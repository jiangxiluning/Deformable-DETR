from typing import List, Tuple, Dict
import copy
import math

import torch
import torch.nn.modules as nn
import torch.nn.functional as F


def generate_ref_points(width: int,
                        height: int):
    grid_y, grid_x = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
    grid_y = grid_y / (height - 1)
    grid_x = grid_x / (width - 1)

    grid = torch.stack((grid_x, grid_y), 2).float()
    grid.requires_grad = False
    return grid


def restore_scale(width: int,
                  height: int,
                  ref_point: torch.Tensor):

    new_point = ref_point.clone().detach()
    new_point[:, :, 0] = new_point[:, :, 0] * (width - 1)
    new_point[:, :, 1] = new_point[:, :, 1] * (height - 1)

    return new_point


class DeformableHeadAttention(nn.Module):
    def __init__(self, h,
                 d_model,
                 k,
                 last_feat_height,
                 last_feat_width,
                 scales=1,
                 dropout=0.1,
                 need_attn=False):
        """
        :param h: number of self attention head
        :param d_model: dimension of model
        :param dropout:
        :param k: number of keys
        """
        super(DeformableHeadAttention, self).__init__()
        assert h == 8  # currently header is fixed 8 in paper
        assert d_model % h == 0
        # We assume d_v always equals d_k, d_q = d_k = d_v = d_m / h
        self.d_k = int(d_model / h)
        self.h = h

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)

        self.scales_hw = []
        for i in range(scales):
            self.scales_hw.append([last_feat_height * 2**i,
                                   last_feat_width * 2**i])

        self.dropout = None
        if self.dropout:
            self.dropout = nn.Dropout(p=dropout)

        self.k = k
        self.scales = scales
        self.last_feat_height = last_feat_height
        self.last_feat_width = last_feat_width

        self.offset_dims = 2*self.h*self.k*self.scales
        self.A_dims = self.h*self.k*self.scales

        # 2MLK for offsets MLK for A_mlqk
        self.offset_proj = nn.Linear(d_model, self.offset_dims)
        self.A_proj = nn.Linear(d_model, self.A_dims)

        self.wm_proj = nn.Linear(d_model, d_model)
        self.need_attn = need_attn
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.offset_proj.weight, 0.0)
        torch.nn.init.constant_(self.A_proj.weight, 0.0)

        torch.nn.init.constant_(self.A_proj.bias, 1/(self.scales * self.k))

        def init_xy(bias, x, y):
            torch.nn.init.constant_(bias[:, 0], float(x))
            torch.nn.init.constant_(bias[:, 1], float(y))

        # caution: offset layout will be  M, L, K, 2
        bias = self.offset_proj.bias.view(self.h, self.scales, self.k, 2)

        init_xy(bias[0], x=-self.k, y=-self.k)
        init_xy(bias[1], x=-self.k, y=0)
        init_xy(bias[2], x=-self.k, y=self.k)
        init_xy(bias[3], x=0, y=-self.k)
        init_xy(bias[4], x=0, y=self.k)
        init_xy(bias[5], x=self.k, y=-self.k)
        init_xy(bias[6], x=self.k, y=0)
        init_xy(bias[7], x=self.k, y=self.k)

    def forward(self,
                query: torch.Tensor,
                keys: List[torch.Tensor],
                ref_point: torch.Tensor,
                mask=None):
        """

        :param query: B, H, W, C
        :param keys: List[B, H, W, C]
        :param ref_point: B, H, W, 2
        :param mask:
        :return:
        """
        # if mask is not None:
        #     mask = mask.unsqueeze(1)
        assert len(keys) == self.scales
        # is_flatten_feat = False

        # W = 1 is flatten
        # if querys[0].size(2) == 1:
        #     is_flatten_feat = True

        # ref_points = []
        # if not is_flatten_feat:
        #     # For encoder, we generate mesh grid
        #     for hw in self.scales_hw:
        #         h, w = hw
        #         ref_points.append(generate_ref_points(width=w, height=h))
        # else:
        #     # For decoder, we predict ref point
        #     object_ref_point = self.ref_point_proj(querys[0])
        #     object_ref_point = F.sigmoid(object_ref_point)
        #     for _ in querys:
        #         # B, H, W, 2
        #         ref_points.append(object_ref_point.clone())

        attns = {'attns': None, 'offsets': None}

        nbatches, H, W, _ = query.shape

        # B, H, W, C
        query = self.q_proj(query)

        # B, H, W, 2MLK
        offset = self.offset_proj(query)
        # B, H, W, M, 2LK
        offset = offset.view(nbatches, H, W, self.h, -1)

        # B, H, W, MLK
        A = self.A_proj(query)
        # # B, H, W, M, LK
        A = A.view(nbatches, H, W, self.h, -1)
        A = F.softmax(A, dim=-1)

        # # B, H, W, M, C_v
        # query = query.view(nbatches, H, W, self.h, self.d_k)

        if self.need_attn:
            attns['attns'] = A
            attns['offsets'] = offset

        offset = offset.view(nbatches, H, W, self.h, self.scales, self.k, 2)
        offset = offset.permute(0, 3, 4, 5, 1, 2, 6).contiguous()
        # B*M, L, K, H, W, 2
        offset = offset.view(nbatches*self.h, self.scales, self.k, H, W, 2)

        A = A.permute(0, 3, 1, 2, 4).contiguous()
        # B*M, H*W, LK
        A = A.view(nbatches*self.h, H*W, -1)

        # # B*M, H*W, LK, 1
        # A = A.unsqueeze(dim=-1)

        # H, W, 2 or  L , 1, 2 for decoder
        # abs_ref_point = restore_scale(width=W,
        #                               height=H,
        #                               ref_point=ref_point)

        scale_features = []
        for l in range(self.scales):
            h, w = self.scales_hw[l]

            # H, W, 2
            reversed_ref_point = restore_scale(height=h, width=w, ref_point=ref_point)

            # 1, H, W, 2
            reversed_ref_point = reversed_ref_point.unsqueeze(dim=0)

            # B, h, w, M, C_v
            scale_feature = self.k_proj(keys[l]).view(nbatches, h, w, self.h, self.d_k)
            # B, M, C_v, h, w
            scale_feature = scale_feature.permute(0, 3, 4, 1, 2).contiguous()
            # B*M, C_v, h, w
            scale_feature = scale_feature.view(-1, self.d_k, h, w)

            k_features = []

            for k in range(self.k):
                points = reversed_ref_point + offset[:, l, k, :, :, :]
                vgrid_x = 2.0 * points[:, :, :, 0] / max(w - 1, 1) - 1.0
                vgrid_y = 2.0 * points[:, :, :, 1] / max(h - 1, 1) - 1.0
                vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)

                # B*M, C_v, H, W
                feat = F.grid_sample(scale_feature, vgrid_scaled, mode='bilinear', padding_mode='zeros')
                k_features.append(feat)
            # B*M, k, C_v, H, W
            k_features = torch.stack(k_features, dim=1)
            scale_features.append(k_features)

        # B*M, L, K, C_v, H, W
        scale_features = torch.stack(scale_features, dim=1)

        # B*M, H*W, C_v, LK
        scale_features = scale_features.permute(0, 4, 5, 3, 1, 2).contiguous()
        scale_features = scale_features.view(nbatches*self.h, H*W, self.d_k, -1)

        # B*M, H*W, C_v
        feat = torch.einsum('nlds, nls -> nld', scale_features, A)

        # B, H, W, C
        feat = feat.view(nbatches, H, W, self.d_k*self.h)
        feat = self.wm_proj(feat)

        if self.dropout:
            feat = self.dropout(feat)

        return feat, attns
