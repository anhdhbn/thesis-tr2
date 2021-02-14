import torch.nn as nn
from copy import deepcopy
from torch import Tensor
from typing import Optional

def getClones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([deepcopy(module) for _ in range(N)])

def with_pos_embed(tensor: Tensor, pos: Optional[Tensor]):
    return tensor if pos is None else tensor + pos