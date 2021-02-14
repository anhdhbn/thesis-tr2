import torch
import torch.nn as nn
from torch import Tensor

from tr2.models.encoder import TransformerEncoder, TransformerEncoderLayer
from tr2.models.decoder import TransformerDecoder, TransformerDecoderLayer

class Transformer(nn.Module):
    def __init__(self,
                hidden_dims=512, 
                num_heads = 8, 
                num_encoder_layer=6, 
                num_decoder_layer=6, 
                dim_feed_forward=2048, 
                dropout=.1
            ):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            hidden_dims=hidden_dims,
            num_heads=num_heads,
            dropout=dropout,
            dim_feedforward=dim_feed_forward
        )
        self.encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layer
        )

        decoder_layer = TransformerDecoderLayer(
            hidden_dims=hidden_dims,
            num_heads=num_heads,
            dropout=dropout,
            dim_feedforward=dim_feed_forward
        )

        decoder_layer2 = TransformerDecoderLayer(
            hidden_dims=hidden_dims,
            num_heads=num_heads,
            dropout=dropout,
            dim_feedforward=dim_feed_forward
        )
        self.decoder = TransformerDecoder(decoder_layer=decoder_layer, num_layers=num_decoder_layer)
        self.decoder2 = TransformerDecoder(decoder_layer=decoder_layer2, num_layers=num_decoder_layer)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, template: Tensor, mask_template: Tensor, pos_template: Tensor, 
                    search: Tensor, mask_search: Tensor, pos_search:Tensor) -> Tensor:
        """
        :param src: tensor of shape [batchSize, hiddenDims, imageHeight // 32, imageWidth // 32]

        :param mask: tensor of shape [batchSize, imageHeight // 32, imageWidth // 32]
                     Please refer to detr.py for more detailed description.

        :param query: object queries, tensor of shape [numQuery, hiddenDims].

        :param pos: positional encoding, the same shape as src.

        :return: tensor of shape [batchSize, num_decoder_layer * WH, hiddenDims]
        """
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = search.shape
        
        template = template.flatten(2).permute(2, 0, 1) # HWxNxC
        search = search.flatten(2).permute(2, 0, 1) # HWxNxC
        
        mask_template = mask_template.flatten(1) # NxHW
        mask_search = mask_search.flatten(1) # NxHW

        pos_template = pos_template.flatten(2).permute(2, 0, 1) # HWxNxC
        pos_search = pos_search.flatten(2).permute(2, 0, 1) # HWxNxC

        memory = self.encoder(template, src_key_padding_mask=mask_template, pos=pos_template, src_mask=mask_template)

        out = self.decoder(search, memory, memory_key_padding_mask=mask_template, pos_template=pos_template, pos_search=pos_search) # num_decoder_layer x WH x N x C 
        out2 = self.decoder2(search, memory, memory_key_padding_mask=mask_template, pos_template=pos_template, pos_search=pos_search) # num_decoder_layer x WH x N x C 
        return out.transpose(1, 2), out2.transpose(1, 2)

def build_transformer(
    hidden_dims=512, 
    num_heads = 8, 
    num_encoder_layer=6, 
    num_decoder_layer=6, 
    dim_feed_forward=2048, 
    dropout=.1
):
    return Transformer(hidden_dims=hidden_dims, 
        num_heads = num_heads, 
        num_encoder_layer = num_encoder_layer, 
        num_decoder_layer = num_decoder_layer, 
        dim_feed_forward = dim_feed_forward, 
        dropout=dropout
    )