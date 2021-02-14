from collections import namedtuple
import torch
from torch.functional import Tensor
import torch.nn as nn
from typing import Optional

from tr2.models.utils import getClones, with_pos_embed

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int):
        super(TransformerEncoder, self).__init__()
        self.layers = getClones(encoder_layer, num_layers)

    def forward(self, 
                src: Tensor, 
                src_mask: Optional[Tensor] = None, 
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None) -> Tensor:
        out = src
        for layer in self.layers:
            out = layer(out, src_mask, src_key_padding_mask, pos)
        return out

class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dims: int, num_heads:int, dropout: float, dim_feedforward: int):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(hidden_dims, num_heads, dropout=dropout)
        self.dropout_sa = nn.Dropout(dropout)
        self.norm_sa = nn.LayerNorm(hidden_dims)

        # Implementation of Feedforward model
        self.dropout_ff = nn.Dropout(dropout)
        self.norm_ff = nn.LayerNorm(hidden_dims)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dims, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dims),
        )

    def forward(self,
                src : Tensor,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        q = k = with_pos_embed(src, pos)

        attn_out, attn_output_weights = self.self_attention(query=q, key=k, value=src, key_padding_mask=src_key_padding_mask)
        x = self.norm_sa(self.dropout_sa(attn_out) + src)

        # Add skip connection, run through normalization and finally dropout
        forward = self.feed_forward(x)
        out = self.norm_ff(self.dropout_ff(forward) + x)
        return out

