from typing import Dict, List, Tuple

import torch
from torch import nn, Tensor
from torch._C import device
from tr2.utils import box_ops
import torch.nn.functional as F

class Tr2Criterion(nn.Module):
    def __init__(self, cls_weight = 1, loc_weight = 1.2, giou_weight = 1.4):
        super(Tr2Criterion, self).__init__()
        self.loc_weight = loc_weight
        self.cls_weight = cls_weight
        self.giou_weight = giou_weight

        self.tr_cls_loss = nn.BCEWithLogitsLoss()

    def forward(self, x: Tuple[Tensor, Tensor], y: Tuple[Tensor, Tensor]) -> Dict[str, Tensor]:
        """
        :param x and y: a tuple containing:
            a tensor of shape [batchSize , 1]
            a tensor of shape [batchSize , 4]

        :return: a dictionary containing classification loss, bbox loss, and gIoU loss
        """
        outputs = {}
        cls, loc = x
        label_cls, label_loc = y
        N, _ = label_loc.shape

        # class loss
        cls_loss = self.tr_cls_loss(cls, label_cls)

        # ignore negative labels
        mask = label_cls != torch.tensor([0], dtype=label_cls.dtype, device=label_cls.device)
        mask = torch.cat((mask, mask, mask, mask), 1)
        loc_mask = loc[mask].view(-1, 4)
        label_loc_mask = label_loc[mask].view(-1, 4)

        if len(label_loc_mask) == 0:
            return None
        # loc loss
        loc_loss = F.mse_loss(loc_mask, label_loc_mask)

        # giou loss
        # giou_loss = 1 - torch.diag(gIoU(loc, label_loc))
        # giou_loss = giou_loss.sum() / (N + 1e-6)

        # iou loss
        iou, _ = box_ops.box_iou(
            box_ops.box_cxcywh_to_xyxy(loc_mask), 
            box_ops.box_cxcywh_to_xyxy(label_loc_mask))
        iou = torch.diagonal(iou)
        iou_loss = torch.mean(1 - iou)

        giou_loss = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(loc_mask),
            box_ops.box_cxcywh_to_xyxy(label_loc_mask)))
        giou_loss = torch.mean(giou_loss)

        outputs['loc_loss'] = loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['giou_loss'] = giou_loss
        outputs['iou_loss'] = iou_loss
        outputs['iou_var'] = torch.std(iou)
        # outputs['total_loss'] = self.cls_weight * cls_loss + self.loc_weight * loc_loss + self.giou_weight * giou_loss
        outputs['total_loss'] = self.cls_weight * cls_loss + self.loc_weight * loc_loss + self.giou_weight * iou_loss
        return outputs

def build_criterion(cls_weight = 1, loc_weight = 1.2, giou_weight = 1.4):
    return Tr2Criterion(cls_weight, loc_weight, giou_weight)