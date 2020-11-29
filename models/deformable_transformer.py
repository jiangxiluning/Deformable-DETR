# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .deformable_attn import DeformableHeadAttention, generate_ref_points


class Transformer(nn.Module):

    def __init__(self,
                 d_model=512,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False,
                 return_intermediate_dec=False,
                 scales=4,
                 k=4,
                 last_height=16,
                 last_width=16, ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model=d_model,
                                                nhead=nhead,
                                                k=k,
                                                scales=scales,
                                                last_feat_height=last_height,
                                                last_feat_width=last_width,
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout,
                                                activation=activation,
                                                normalize_before=normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model=d_model,
                                                nhead=nhead,
                                                k=k,
                                                scales=scales,
                                                last_feat_height=last_height,
                                                last_feat_width=last_width,
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout,
                                                activation=activation,
                                                normalize_before=normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.query_ref_point_proj = nn.Linear(d_model, 2)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: List[Tensor],
                masks: List[Tensor],
                query_embed,
                pos_embeds: List[Tensor]):
        bs = src[0].size(0)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        # B, C H, W -> B, H, W, C
        for index in range(len(src)):
            src[index] = src[index].permute(0, 2, 3, 1)
            pos_embeds[index] = pos_embeds[index].permute(0, 2, 3, 1)

        # B, H, W, C
        ref_points = []
        for tensor in src:
            _, height, width, _ = tensor.shape
            ref_point = generate_ref_points(width=width,
                                            height=height)
            ref_point = ref_point.type_as(src[0])
            # H, W, 2 -> B, H, W, 2
            ref_point = ref_point.unsqueeze(0).repeat(bs, 1, 1, 1)
            ref_points.append(ref_point)

        tgt = torch.zeros_like(query_embed)

        # List[B, H, W, C]
        memory = self.encoder(src,
                              ref_points,
                              src_key_padding_masks=masks,
                              poses=pos_embeds)

        # L, B, C
        query_ref_point = self.query_ref_point_proj(tgt)
        query_ref_point = F.sigmoid(query_ref_point)

        # Decoder Layers, L, B ,C
        hs = self.decoder(tgt, memory,
                          query_ref_point,
                          memory_key_padding_masks=masks,
                          poses=pos_embeds,
                          query_pos=query_embed)

        return hs, query_ref_point, memory,


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self,
                src_tensors: List[Tensor],
                ref_points: List[Tensor],
                src_masks: Optional[List[Tensor]] = None,
                src_key_padding_masks: Optional[List[Tensor]] = None,
                poses: Optional[List[Tensor]] = None):
        outputs = src_tensors

        for layer in self.layers:
            outputs = layer(outputs,
                            ref_points,
                            src_masks=src_masks,
                            src_key_padding_masks=src_key_padding_masks,
                            poses=poses)

        if self.norm is not None:
            for index, output in enumerate(outputs):
                outputs[index] = self.norm(output)

        return outputs


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                ref_point: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_masks: Optional[List[Tensor]] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_masks: Optional[List[Tensor]] = None,
                poses: Optional[List[Tensor]] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output,
                           memory,
                           ref_point,
                           tgt_mask=tgt_mask,
                           memory_masks=memory_masks,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_masks=memory_key_padding_masks,
                           poses=poses, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self,
                 d_model: int,
                 nhead: int,
                 k: int,
                 scales: int,
                 last_feat_height: int,
                 last_feat_width: int,
                 need_attn: bool = False,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.ms_deformbale_attn = DeformableHeadAttention(h=nhead,
                                                          d_model=d_model,
                                                          k=k,
                                                          scales=scales,
                                                          last_feat_height=last_feat_height,
                                                          last_feat_width=last_feat_width,
                                                          dropout=dropout,
                                                          need_attn=need_attn)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.need_attn = need_attn
        self.attns = []

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src_tensors: List[Tensor],
                     ref_points: List[Tensor],
                     src_masks: Optional[List[Tensor]] = None,
                     src_key_padding_masks: Optional[List[Tensor]] = None,
                     poses: Optional[List[Tensor]] = None):
        if src_masks is None:
            src_masks = [None] * len(src_tensors)

        if src_key_padding_masks is None:
            src_key_padding_masks = [None] * len(src_tensors)

        if poses is None:
            poses = [None] * len(poses)

        feats = []
        src_tensors = [self.with_pos_embed(tensor, pos) for tensor, pos in zip(src_tensors, poses)]
        for src, ref_point, _, src_key_padding_mask, pos in zip(src_tensors,
                                                                ref_points,
                                                                src_masks,
                                                                src_key_padding_masks,
                                                                poses):
            # src = self.with_pos_embed(src, pos)
            src2, attns = self.ms_deformbale_attn(src,
                                                  src_tensors,
                                                  ref_point,
                                                  query_mask=src_key_padding_mask,
                                                  key_masks=src_key_padding_masks)

            if self.need_attn:
                self.attns.append(attns)

            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
            feats.append(src)

        return feats

    def forward_pre(self,
                    src_tensors: List[Tensor],
                    ref_points: List[Tensor],
                    src_masks: Optional[List[Tensor]] = None,
                    src_key_padding_masks: Optional[List[Tensor]] = None,
                    poses: Optional[List[Tensor]] = None):

        if src_masks is None:
            src_masks = [None] * len(src_tensors)

        if src_key_padding_masks is None:
            src_key_padding_masks = [None] * len(src_tensors)

        if poses is None:
            poses = [None] * len(src_tensors)

        feats = []

        src_tensors = [self.with_pos_embed(tensor, pos) for tensor, pos in zip(src_tensors, poses)]
        for src, ref_point, _, src_key_padding_mask, pos in zip(src_tensors,
                                                                ref_points,
                                                                src_masks,
                                                                src_key_padding_masks,
                                                                poses):
            src2 = self.norm1(src, pos)
            # src2 = self.with_pos_embed(src2, pos)
            src2, attns = self.ms_deformbale_attn(src2, src_tensors, ref_point, query_mask=src_key_padding_masks)

            if self.need_attn:
                self.attns.append(attns)

            src = src + self.dropout1(src2)
            src2 = self.norm2(src)

            src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
            src = src + self.dropout2(src2)
            feats.append(src)

        return feats

    def forward(self,
                src_tensors: List[Tensor],
                ref_points: List[Tensor],
                src_masks: Optional[List[Tensor]] = None,
                src_key_padding_masks: Optional[List[Tensor]] = None,
                poses: Optional[List[Tensor]] = None):
        if self.normalize_before:
            return self.forward_pre(src_tensors,
                                    ref_points,
                                    src_masks,
                                    src_key_padding_masks,
                                    poses)
        return self.forward_post(src_tensors,
                                 ref_points,
                                 src_masks,
                                 src_key_padding_masks,
                                 poses)


class TransformerDecoderLayer(nn.Module):

    def __init__(self,
                 d_model: int,
                 nhead: int,
                 k: int,
                 scales: int,
                 last_feat_height: int,
                 last_feat_width: int,
                 need_attn: bool = False,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.ms_deformbale_attn = DeformableHeadAttention(h=nhead,
                                                          d_model=d_model,
                                                          k=k,
                                                          scales=scales,
                                                          last_feat_height=last_feat_height,
                                                          last_feat_width=last_feat_width,
                                                          dropout=dropout,
                                                          need_attn=need_attn)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.need_attn = need_attn
        self.attns = []

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     ref_point: Tensor,
                     tgt_mask: Optional[Tensor] = None,
                     memory_masks: Optional[List[Tensor]] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_masks: Optional[List[Tensor]] = None,
                     poses: Optional[List[Tensor]] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        memory = [self.with_pos_embed(tensor, pos) for tensor, pos in zip(memory, poses)]

        # L, B, C -> B, L, 1, C
        tgt = tgt.transpose(0, 1).unsqueeze(dim=2)
        ref_point = ref_point.transpose(0, 1).unsqueeze(dim=2)

        # B, L, 1, C
        tgt2, attns = self.ms_deformbale_attn(tgt,
                                              memory,
                                              ref_point,
                                              query_mask=None,
                                              key_masks=memory_key_padding_masks)

        if self.need_attn:
            self.attns.append(attns)

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        # B, L, 1, C -> L, B, C
        tgt = tgt.squeeze(dim=2)
        tgt = tgt.transpose(0, 1).contiguous()

        # decoder we only one query tensor
        return tgt

    def forward_pre(self, tgt: Tensor,
                    memory: List[Tensor],
                    ref_point: Tensor,
                    tgt_mask: Optional[Tensor] = None,
                    memory_masks: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_masks: Optional[Tensor] = None,
                    poses: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)

        memory = [self.with_pos_embed(tensor, pos) for tensor, pos in zip(memory, poses)]

        # L, B, C -> B, L, 1, C
        tgt2 = tgt2.transpose(0, 1).unsqueeze(dim=2)
        ref_point = ref_point.transpose(0, 1).unsqueeze(dim=2)

        # B, L, 1, 2
        tgt2, attns = self.ms_deformbale_attn(tgt2, memory, ref_point,
                                              query_mask=None,
                                              key_masks=memory_key_padding_masks)
        if self.need_attn:
            self.attns.append(attns)

        # B, L, 1, C -> L, B, C
        tgt2 = tgt2.squeeze(dim=2)
        tgt2 = tgt2.transpose(0, 1).contiguous()

        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt

    def forward(self, tgt: Tensor,
                memory: List[Tensor],
                ref_point: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_masks: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_masks: Optional[Tensor] = None,
                poses: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, ref_point, tgt_mask, memory_masks,
                                    tgt_key_padding_mask, memory_key_padding_masks, poses, query_pos)
        return self.forward_post(tgt, memory, ref_point, tgt_mask, memory_masks,
                                 tgt_key_padding_mask, memory_key_padding_masks, poses, query_pos)


def _get_clones(module, num):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        scales=args.scales,
        k=args.k,
        last_height=args.last_height,
        last_width=args.last_width
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
