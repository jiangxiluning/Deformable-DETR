# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

import torch

from models.matcher import HungarianMatcher
from models.position_encoding import PositionEmbeddingSine, PositionEmbeddingLearned
from models.backbone import Backbone
from util import box_ops
from util.misc import nested_tensor_from_tensor_list
from hubconf import detr_resnet50, detr_resnet50_panoptic
from models.deformable_attn import DeformableHeadAttention, generate_ref_points


class Tester(unittest.TestCase):

    def test_box_cxcywh_to_xyxy(self):
        t = torch.rand(10, 4)
        r = box_ops.box_xyxy_to_cxcywh(box_ops.box_cxcywh_to_xyxy(t))
        self.assertLess((t - r).abs().max(), 1e-5)

    @staticmethod
    def indices_torch2python(indices):
        return [(i.tolist(), j.tolist()) for i, j in indices]

    def test_hungarian(self):
        n_queries, n_targets, n_classes = 100, 15, 91
        logits = torch.rand(1, n_queries, n_classes + 1)
        boxes = torch.rand(1, n_queries, 4)
        tgt_labels = torch.randint(high=n_classes, size=(n_targets,))
        tgt_boxes = torch.rand(n_targets, 4)
        matcher = HungarianMatcher()
        targets = [{'labels': tgt_labels, 'boxes': tgt_boxes}]
        indices_single = matcher({'pred_logits': logits, 'pred_boxes': boxes}, targets)
        indices_batched = matcher({'pred_logits': logits.repeat(2, 1, 1),
                                   'pred_boxes': boxes.repeat(2, 1, 1)}, targets * 2)
        self.assertEqual(len(indices_single[0][0]), n_targets)
        self.assertEqual(len(indices_single[0][1]), n_targets)
        self.assertEqual(self.indices_torch2python(indices_single),
                         self.indices_torch2python([indices_batched[0]]))
        self.assertEqual(self.indices_torch2python(indices_single),
                         self.indices_torch2python([indices_batched[1]]))

        # test with empty targets
        tgt_labels_empty = torch.randint(high=n_classes, size=(0,))
        tgt_boxes_empty = torch.rand(0, 4)
        targets_empty = [{'labels': tgt_labels_empty, 'boxes': tgt_boxes_empty}]
        indices = matcher({'pred_logits': logits.repeat(2, 1, 1),
                           'pred_boxes': boxes.repeat(2, 1, 1)}, targets + targets_empty)
        self.assertEqual(len(indices[1][0]), 0)
        indices = matcher({'pred_logits': logits.repeat(2, 1, 1),
                           'pred_boxes': boxes.repeat(2, 1, 1)}, targets_empty * 2)
        self.assertEqual(len(indices[0][0]), 0)

    def test_position_encoding_script(self):
        m1, m2 = PositionEmbeddingSine(), PositionEmbeddingLearned()
        mm1, mm2 = torch.jit.script(m1), torch.jit.script(m2)  # noqa

    def test_backbone_script(self):
        backbone = Backbone('resnet50', True, False, False)
        torch.jit.script(backbone)  # noqa

    def test_model_script_detection(self):
        model = detr_resnet50(pretrained=False).eval()
        scripted_model = torch.jit.script(model)
        x = nested_tensor_from_tensor_list([torch.rand(3, 200, 200), torch.rand(3, 200, 250)])
        out = model(x)
        out_script = scripted_model(x)
        self.assertTrue(out["pred_logits"].equal(out_script["pred_logits"]))
        self.assertTrue(out["pred_boxes"].equal(out_script["pred_boxes"]))

    def test_model_script_panoptic(self):
        model = detr_resnet50_panoptic(pretrained=False).eval()
        scripted_model = torch.jit.script(model)
        x = nested_tensor_from_tensor_list([torch.rand(3, 200, 200), torch.rand(3, 200, 250)])
        out = model(x)
        out_script = scripted_model(x)
        self.assertTrue(out["pred_logits"].equal(out_script["pred_logits"]))
        self.assertTrue(out["pred_boxes"].equal(out_script["pred_boxes"]))
        self.assertTrue(out["pred_masks"].equal(out_script["pred_masks"]))

    def test_model_detection_different_inputs(self):
        model = detr_resnet50(pretrained=False).eval()
        # support NestedTensor
        x = nested_tensor_from_tensor_list([torch.rand(3, 200, 200), torch.rand(3, 200, 250)])
        out = model(x)
        self.assertIn('pred_logits', out)
        # and 4d Tensor
        x = torch.rand(1, 3, 200, 200)
        out = model(x)
        self.assertIn('pred_logits', out)
        # and List[Tensor[C, H, W]]
        x = torch.rand(3, 200, 200)
        out = model([x])
        self.assertIn('pred_logits', out)

    def test_deformable_attn(self):
        defomable_attn = DeformableHeadAttention(h=8,
                                                 d_model=256,
                                                 k=4,
                                                 last_feat_width=16,
                                                 last_feat_height=16,
                                                 scales=4,
                                                 need_attn=True)
        defomable_attn = defomable_attn.cuda()
        w = 16
        h = 16
        querys = []
        ref_points = []
        for i in range(4):
            ww = w * 2**i
            hh = h * 2**i
            q = torch.rand([2, hh, ww, 256])
            q = q.cuda()
            querys.append(q)
            ref_point = generate_ref_points(width=ww, height=hh)
            ref_point = ref_point.type_as(q)
            ref_points.append(ref_point)

        feat, ref_points, attns = defomable_attn(querys[0], querys, ref_points[0])
        self.assertTrue(True)

    def test_backbone_forward(self):
        backbone = Backbone('resnet50', True, True, False)
        x = nested_tensor_from_tensor_list([torch.rand(3, 200, 200), torch.rand(3, 200, 250)])

        out = backbone(x)
        for key, value in out.items():
            print('{} {}'.format(key, value.tensors.shape))

    def test_transformer_forward(self):
        backbone = Backbone('resnet50', True, True, False)
        x = nested_tensor_from_tensor_list([torch.rand(3, 200, 200), torch.rand(3, 200, 250)])

        out = backbone(x)
        for key, value in out.items():
            print('{} {}'.format(key, value.tensors.shape))


if __name__ == '__main__':
    unittest.main()
