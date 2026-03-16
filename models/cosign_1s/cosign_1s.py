"""This module contains code for our implementation of 1-stream CoSign architecture."""

import torch.nn as nn

from models.stgcn.models import STGCNGraphConv as STGCN


class STGCNArgs:
    def __init__(self, Kt, Ks, act_func, graph_conv_type, gso, enable_bias, droprate):
        self.Kt = Kt  # Temporal Kernel Size
        self.Ks = Ks  # Spatial Kernel Size
        self.act_func = act_func  # activation funciton
        self.graph_conv_type = graph_conv_type
        self.gso = gso  # graph signal operator - adjecency matrix
        self.enable_bias = enable_bias
        self.droprate = droprate


cs_blocks = [
    [3],  # input
    [64, 64, 64],  # layer 1
    [64, 64, 64],  # layer 2
    [64, 64, 64],  # layer 3
]

hand_args = STGCNArgs()


class CoSign1s(nn.Module):
    def __init__(self):
        super().__init__(self)

        self.groupspecific_gcn = nn.ModuleDict(
            {
                "face": STGCN(blocks=cs_blocks),
                "mouth": STGCN(blocks=cs_blocks),
                "body": STGCN(blocks=cs_blocks),
                "hands": STGCN(blocks=cs_blocks),
            }
        )
