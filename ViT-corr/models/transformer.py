# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COTR/DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from turtle import forward
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from COTR.utils import debug_utils


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # def forward(self, src, mask, query_embed, pos_embed):
    def forward(self, query_image, key_image, query_mask, queries, query_position, key_position):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = query_image.shape
        query_image = query_image.flatten(2).permute(2, 0, 1)
        key_image = key_image.flatten(2).permute(2, 0, 1)
        query_pos_embed = query_position.flatten(2).permute(2, 0, 1)
        key_pos_embed = key_position.flatten(2).permute(2, 0, 1)
        # print(" In Transformer Model")
        # print(" Query and Key Projection Shapes = ", query_image.shape, key_image.shape)
        # print(" Query and Key Position Shapes = ", query_position.shape, key_position.shape)
        # print(" Query and Key Position Embedding Shapes = ", query_pos_embed.shape, key_pos_embed.shape)

        query_mask = query_mask.flatten(1)
        tgt = torch.zeros_like(queries)
        memory = self.encoder(query_image, key_image, query_mask, query_pos_embed, key_pos_embed)
        hs = self.decoder(tgt, memory, query_mask, queries, query_pos_embed, key_pos_embed)
                        
        # print("Src Shape = ", src.shape)
        # print("Mask Shape = ", mask.shape)
        # print("Query Pos Shape = ", query_embed.shape)
        # bs, c, h, w = src.shape
        # src = src.flatten(2).permute(2, 0, 1)
        # pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        # mask = mask.flatten(1)
        # print("src shape after = ", src.shape)
        # print("pos shape after = ", pos_embed.shape)
        # print("mask shape after = ", mask.shape)

        # tgt = torch.zeros_like(query_embed)
        # memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        # hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
        #                   pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    # def forward(self, src,
    #             mask: Optional[Tensor] = None,
    #             src_key_padding_mask: Optional[Tensor] = None,
    #             pos: Optional[Tensor] = None):
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def forward(self, query_image, key_image, query_mask, query_pos_embed, key_pos_embed):
        # output = src
        value = self.with_pos_embed(torch.zeros_like(key_image), key_pos_embed)

        for layer in self.layers:
            # output = layer(output, src_mask=mask,
            #                src_key_padding_mask=src_key_padding_mask, pos=pos)
            output = layer(value, query_image, key_image, query_mask, query_pos_embed, key_pos_embed)
            value += output

        return value


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
    
    def forward(self, tgt, memory, query_mask, queries, query_pos_embed, key_pos_embed):
        decoder_value = tgt
        intermediate = []
        for layer in self.layers:
            decoder_value = layer(decoder_value, memory, query_mask, queries, query_pos_embed, key_pos_embed)
            if self.return_intermediate:
                intermediate.append(decoder_value)

        if self.norm is not None:
            decoder_value = self.norm(decoder_value)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(decoder_value)
      
        if self.return_intermediate:
            return torch.stack(intermediate)

        return decoder_value.unsqueeze(0)
    
    # def forward(self, tgt, memory,
    #             tgt_mask: Optional[Tensor] = None,
    #             memory_mask: Optional[Tensor] = None,
    #             tgt_key_padding_mask: Optional[Tensor] = None,
    #             memory_key_padding_mask: Optional[Tensor] = None,
    #             pos: Optional[Tensor] = None,
    #             query_pos: Optional[Tensor] = None):

        # output = tgt

        # intermediate = []
        # for layer in self.layers:
        #     output = layer(output, memory, tgt_mask=tgt_mask,
        #                    memory_mask=memory_mask,
        #                    tgt_key_padding_mask=tgt_key_padding_mask,
        #                    memory_key_padding_mask=memory_key_padding_mask,
        #                    pos=pos, query_pos=query_pos)
        #     if self.return_intermediate:
        #         intermediate.append(self.norm(output))

        # if self.norm is not None:
        #     output = self.norm(output)
        #     if self.return_intermediate:
        #         intermediate.pop()
        #         intermediate.append(output)

        # if self.return_intermediate:
        #     return torch.stack(intermediate)

        # return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    # def forward(self,
    #             src,
    #             src_mask: Optional[Tensor] = None,
    #             src_key_padding_mask: Optional[Tensor] = None,
    #             pos: Optional[Tensor] = None):
    def forward(self, value, query_image, key_image, query_mask, query_pos_embed, key_pos_embed):
        # q = k = self.with_pos_embed(src, pos)
        # src2 = self.self_attn(query=q,
        #                       key=k,
        #                       value=src,
        #                       attn_mask=src_mask,
        #                       key_padding_mask=src_key_padding_mask)[0]
        # src = src + self.dropout1(src2)
        # src = self.norm1(src)
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # src = src + self.dropout2(src2)
        # src = self.norm2(src)
        # return src
        query = self.with_pos_embed(query_image, query_pos_embed)
        key = self.with_pos_embed(key_image, key_pos_embed)
        layer_out = self.self_attn(query = query, key = key, value = value, key_padding_mask = query_mask)[0]
        value = value.clone() + self.dropout1(layer_out)
        value = self.norm1(value)
        layer_out = self.linear2(self.dropout(self.activation(self.linear1(value))))
        value = value.clone() + self.dropout2(layer_out)
        value = self.norm2(value)
        return value


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
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

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, value, memory, query_mask, queries, query_pos_embed, key_pos_embed):
        layer_out = self.multihead_attn(query = self.with_pos_embed(value, queries), key = self.with_pos_embed(memory, query_pos_embed), value = memory, key_padding_mask = query_mask)[0]
        value = value.clone() + self.dropout2(layer_out)
        value = self.norm2(value)
        layer_out = self.linear2(self.dropout(self.activation(self.linear1(value))))
        value = value.clone() + self.dropout3(layer_out)
        value = self.norm3(value)
        return value


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        return_intermediate_dec=True,
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
