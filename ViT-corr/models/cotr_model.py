import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from COTR.utils import debug_utils, constants, utils
from .misc import (NestedTensor, nested_tensor_from_tensor_list)
from .backbone import build_backbone
from .transformer import build_transformer
from .position_encoding import NerfPositionalEncoding, MLP

class COTR(nn.Module):

    def __init__(self, backbone, transformer, sine_type='lin_sine'):
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.corr_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.query_proj = NerfPositionalEncoding(hidden_dim // 4, sine_type)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone

    def forward(self, samples: NestedTensor, queries, mode):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        
        query_image_tensor = []
        query_mask_tensor = []
        key_image_tensor = []
        key_mask_tensor = []

        #print("Batch - ", len(samples.tensors))
        for i in range(len(samples.tensors)):
            # query_image, query_position = self.backbone(NestedTensor(samples.tensors[i][:, :, 0:256].unsqueeze(0), samples.mask[i][:, 0:256].unsqueeze(0)))
            # key_image, key_position = self.backbone(NestedTensor(samples.tensors[i][:, :, 256:512].unsqueeze(0), samples.mask[i][:, 256:512].unsqueeze(0)))
            query_image = samples.tensors[i][:, :, 0:256].unsqueeze(0)
            query_mask = samples.mask[i][:, 0:256].unsqueeze(0)
            key_image = samples.tensors[i][:, :, 256:512].unsqueeze(0)
            key_mask = samples.mask[i][:, 256:512].unsqueeze(0)
            query_image_tensor.append(query_image)
            key_image_tensor.append(key_image)
            query_mask_tensor.append(query_mask)
            key_mask_tensor.append(key_mask)

        query_image = torch.cat(query_image_tensor, dim=0)
        key_image = torch.cat(key_image_tensor, dim=0)
        query_mask = torch.cat(query_mask_tensor, dim=0)
        key_mask = torch.cat(key_mask_tensor, dim=0)

        # print("Before backbone")
        # print("query_image: ", query_image.shape)
        # print("key_image: ", key_image.shape)
        # print("query mask: ", query_mask.shape)
        query_image_out, query_position = self.backbone(NestedTensor(query_image, query_mask))
        key_image_out, key_position = self.backbone(NestedTensor(key_image, key_mask))
        query_image = query_image_out[0].tensors
        key_image = key_image_out[0].tensors
        query_position = query_position[0]
        key_position = key_position[0]
        query_mask = query_image_out[0].mask
        key_mask = key_image_out[0].mask
        # print("After backbone")
        # print("query_image: ", query_image.shape)
        # print("key_image: ", key_image.shape)
        # print("query mask: ", query_mask.shape)
        
        assert query_mask is not None
        _b, _q, _ = queries.shape
        queries = queries.reshape(-1, 2)
        queries = self.query_proj(queries).reshape(_b, _q, -1)
        queries = queries.permute(1, 0, 2)
        
        #print(" *** Shapes of all inputs - In COTR_MODEL *** ")
        #print("QUERY - ", queries.shape)
        #print("QUERY IMAGE - ", query_image.shape)
        #print("QUERY POSITION - ", query_position.shape)
        #print("QUERY MASK - ", query_mask.shape)
        #print("KEY IMAGE - ", key_image.shape)
        #print("KEY POSITION - ", key_position.shape)
        #print("KEY MASK - ", mask.shape)
        
        if mode == "normal":
            hs = self.transformer(self.input_proj(query_image), self.input_proj(key_image), query_mask, queries, query_position, key_position)[0]
        else:
            hs = self.transformer(self.input_proj(key_image), self.input_proj(query_image), key_mask, queries, key_position, query_position)[0]
        outputs_corr = self.corr_embed(hs)
        out = {'pred_corrs': outputs_corr}
        
        # print(" *** Shapes of all outputs - In COTR_MODEL *** ")
        # print("hs - ", hs.shape)
        # print("OUTPUTS_CORR - ", outputs_corr.shape)
        # print(" *** End of shapes *** ")
        return out


def build(args):
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    model = COTR(
        backbone,
        transformer,
        sine_type=args.position_embedding,
    )
    return model
