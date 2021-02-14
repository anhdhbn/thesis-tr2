
from torchvision.models._utils import IntermediateLayerGetter
import torchvision
import torch.nn as nn
from torch import Tensor

from typing import Tuple, Dict, List
import torch.nn.functional as F
import torch

from tr2.core.config import cfg
from tr2.models.embedding import PositionEmbeddingSine
from tr2.utils.misc import NestedTensor

class Backbone(nn.Module):
    def __init__(self, backbone_name:str="resnet50"):
        super().__init__()
        backbone = getattr(torchvision.models, backbone_name)(pretrained=True)
        for name, parameter in backbone.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        # return_layers = {"layer1": "layer1", "layer2": "layer2", "layer3": "layer3", "layer4": "layer4"}
        return_layers = {"layer4": "layer4"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        for k, v in self.body.items():
            setattr(self, k, v)
        self.num_channels = 512 if backbone_name in ('resnet18', 'resnet34') else 2048
    
    def forward(self, tensor_list: NestedTensor) -> Tensor:
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

class Joiner(nn.Sequential):
    def __init__(self, backbone: nn.Module, position_embedding: nn.Module):
        super(Joiner, self).__init__(backbone, position_embedding)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        features = self.backbone(x)
        return features, self.position_embedding(features)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos

def build_backbone(hidden_dims):
    position_embedding = PositionEmbeddingSine(num_pos_feats=hidden_dims//2, normalize=True)
    backbone = Backbone(cfg.BACKBONE.TYPE)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model