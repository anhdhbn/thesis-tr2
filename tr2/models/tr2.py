from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch

from tr2.core.config import cfg
from tr2.models.transformer import Transformer
from tr2.models.backbone import build_backbone
from tr2.models.transformer import build_transformer
from tr2.models.criterion import build_criterion

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class Tr2(nn.Module):
    def __init__(self,
                backbone: nn.Module,
                transformer: nn.Module,
                hidden_dims: int):
        super().__init__()
        self.backbone = backbone
        self.reshape = nn.Conv2d(backbone.num_channels, hidden_dims, 1)
        self.transformer = transformer
        self.adap = nn.AdaptiveAvgPool2d((None, 1))

        self.class_embed = nn.Linear(hidden_dims, 1)
        self.bbox_embed = MLP(input_dim=hidden_dims, hidden_dim=hidden_dims, output_dim=4, num_layers=3)
    
    def forward(self, template: Tensor, search: Tensor):
        template, pos_template = self.backbone(template)
        template, mask_template = template[-1].decompose()
        template = self.reshape(template)
        pos_template = pos_template[-1]
        assert mask_template is not None


        search, pos_search = self.backbone(search)
        search, mask_search = search[-1].decompose()
        search = self.reshape(search)
        pos_search = pos_search[-1]
        assert mask_search is not None

        out, out2 = self.transformer(template, mask_template, pos_template, search, mask_search, pos_search)

        # [6, 32, 16, 512]
        out = self.adap(out[-1].transpose(1,2)).flatten(1)
        out2 = self.adap(out2[-1].transpose(1,2)).flatten(1)

        outputs_class = self.class_embed(out)
        outputs_coord = self.bbox_embed(out2).sigmoid()
        
        return outputs_class, outputs_coord

def build_tr2():
    hidden_dims=cfg.TRANSFORMER.KWARGS['hidden_dims']
    backbone = build_backbone(hidden_dims)
    transformer = build_transformer(**cfg.TRANSFORMER.KWARGS)
    tr2 = Tr2(
        backbone,
        transformer,
        hidden_dims=hidden_dims
    )

    criterion = build_criterion(
        **cfg.TRAIN.WEIGHT
    )
    return tr2, criterion