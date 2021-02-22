from collections import namedtuple
import torch
from torch.functional import Tensor
import torch.nn as nn
from typing import Optional

from tr2.models.utils import getClones, with_pos_embed

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer: nn.Module, num_layers: int):
        super(TransformerDecoder, self).__init__()
        self.layers = getClones(decoder_layer, num_layers)

    def forward(self, 
                search: Tensor, 
                memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos_template: Optional[Tensor] = None,
                pos_search: Optional[Tensor] = None):
        out = search

        intermediate = []

        for layer in self.layers:
            out = layer(
                search=search,
                memory = memory,
                tgt_mask = tgt_mask,
                memory_mask = memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask = memory_key_padding_mask,
                pos_template = pos_template,
                pos_search = pos_search
                )
            intermediate.append(out)

        return torch.stack(intermediate)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_dims: int, num_heads:int, dropout: float, dim_feedforward: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dims, num_heads=num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dims)

        self.attn = nn.MultiheadAttention(hidden_dims, num_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dims)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dims, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dims),
            nn.Dropout(dropout),
        )

        self.norm_ff = nn.LayerNorm(hidden_dims)

    def forward(self, 
                search: Tensor, 
                memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos_template: Optional[Tensor] = None,
                pos_search: Optional[Tensor] = None):
        q = k = with_pos_embed(search, pos_search)
        # self-att
        search2, _ = self.self_attn(query=q, key=k, value=search, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)
        
        search = search + self.dropout1(search2)
        search = self.norm1(search)

        # query from template
        # key, value from search
        q2 = with_pos_embed(memory, pos_template)
        template_att2, _ = self.attn(query=q2,
                                   key=k,
                                   value=search, attn_mask=memory_mask,
                                   key_padding_mask=tgt_key_padding_mask)
        
        # add norm + dropout
        template_att = q2 + self.dropout2(template_att2)
        template_att = self.norm2(template_att)

        # feed forward + norm + dropout
        template_att2 = self.feed_forward(template_att)
        template_att = template_att + template_att2
        template_att = self.norm_ff(template_att)
        return template_att


