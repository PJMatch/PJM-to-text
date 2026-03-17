"""This module contains code for our implementation of 1-stream CoSign architecture."""

import numpy as np
import torch
import torch.nn as nn

from models.cosign_1s.gso import GSOGenerator
from models.stgcn.models import STGCNGraphConv as STGCN

CONFIG_FILE_PATH = "./config.json"
CONST_KS = 2  # DO NOT CHANGE

COSIGN_BLOCKS = [
    [3],  # input
    [64, 64, 64],  # layer 1
    [64, 64, 64],  # layer 2
    [64, 64, 64],  # layer 3
]


class STGCNArgs:
    """Class that holds specific STGCN configuration arguments.

    Later the object of this class is passed to the constructor of STGCN.
    """

    def __init__(self, Kt, Ks, act_func, graph_conv_type, gso, enable_bias, droprate):
        """STGCNArgs constructor."""
        self.Kt = Kt  # Temporal Kernel Size
        self.Ks = Ks  # Spatial Kernel Size
        self.act_func = act_func  # activation funciton, can be glu / gtu / relu / silu
        self.graph_conv_type = graph_conv_type  # can be 'cheb_graph_conv' or 'graph_conv'
        self.gso = gso  # graph signal operator - adjecency matrix

        self.enable_bias = enable_bias  # bool
        self.droprate = droprate  # value p for nn.Dropout in the STConvBlock


class STGCNCoSign1s(nn.Module):
    """CoSign-1s' ST-GCN implementation.

    Based on:
    @InProceedings{Jiao_2023_ICCV,
        author    = {Jiao, Peiqi and Min, Yuecong and Li, Yanan and Wang, Xiaotao and Lei, Lei and
                    Chen, Xilin},
        title     = {CoSign: Exploring Co-occurrence Signals in Skeleton-based Continuous Sign
                    Language Recognition},
        booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision
                    (ICCV)},
        month     = {October},
        year      = {2023},
        pages     = {20676-20686}
    }
    """

    def __init__(self, config_path="./config.json"):
        """CoSign1s ST-GCN's constructor."""
        super().__init__()

        self.gso_generator = GSOGenerator(config_path)
        self.config = self.gso_generator.config

        self.gcn_out_dim = COSIGN_BLOCKS[-1][-1]

        # TODO: check the actual LSTM input size and put it here
        self.fusion_out_dim = 512  # this needs to be the size of LSTM input

        self.offsets = {
            "body": 0,
            "face": 33,
            "mouth": 33,  # mouth is part of the face detection so the same offset
            "l_hand": 511,
            "r_hand": 532,
        }

        self.gcn_modules = nn.ModuleDict(
            {
                "face": STGCN(
                    args=self._create_args(self.gso_generator.gsos["face"]),
                    blocks=COSIGN_BLOCKS,
                    n_vertex=len(self.config["face"]),
                ),
                "mouth": STGCN(
                    args=self._create_args(self.gso_generator.gsos["mouth"]),
                    blocks=COSIGN_BLOCKS,
                    n_vertex=len(self.config["mouth"]),
                ),
                "body": STGCN(
                    args=self._create_args(self.gso_generator.gsos["body"]),
                    blocks=COSIGN_BLOCKS,
                    n_vertex=len(self.config["body"]),
                ),
                # both hands share the same weights - one module for both
                "hands": STGCN(
                    args=self._create_args(self.gso_generator.gsos["hands"]),
                    blocks=COSIGN_BLOCKS,
                    n_vertex=len(self.config["hands"]),
                ),
            }
        )

        num_groups_for_fusion = 5  # each per group (face + body + 2 * hand + mouth)
        self.fusion_in_dim = num_groups_for_fusion * self.gcn_out_dim

        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(self.fusion_in_dim, self.fusion_out_dim, kernel_size=1),
            nn.BatchNorm1d(self.fusion_out_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )  # TODO: in CoSign paper they say smth about Bernouli distribution for dropout - eqn. (4)

    def forward(self, x):
        """Forward function of CoSign1s ST-GCN module.

        Args:
            x :[Batch, Channels, Timesteps, Vertices] -> [B, 3, T, 553 (or less)]
                x is original, all-point vector from the .npy files

        Returns:
            v_fused : # frame-wise feature for LSTM
        """
        centralized_groups = {}
        for name in ["body", "face", "mouth", "l_hand", "r_hand"]:
            local_indices = np.array(self.config[name])

            global_indices = local_indices + self.offsets[name]

            group_data = x[:, :, :, global_indices]  # [B, 3, T, V_local]

            # first point in each group MUST be the root
            root_point = group_data[:, :, :, 0].unsqueeze(-1)
            # centralization - eqn. (2)
            centralized_groups[name] = group_data - root_point

        features = []

        features.append(self.gcn_modules["body"](centralized_groups["body"]))
        features.append(self.gcn_modules["face"](centralized_groups["face"]))
        features.append(self.gcn_modules["mouth"](centralized_groups["mouth"]))
        features.append(self.gcn_modules["hands"](centralized_groups["l_hand"]))
        features.append(self.gcn_modules["hands"](centralized_groups["r_hand"]))

        v = torch.cat(features, dim=-1)  # [B, T, 320]

        # TODO: if we implement dropout mask it will be here
        # v = v * xi

        # [B, T, 5*C] -> [B, 5*C, T] for BatchNorm1d in MLP
        v = v.permute(0, 2, 1)
        v_fused = self.fusion_mlp(v)

        return v_fused

    # STGCNArgs parameter selection:
    # Kt - in original ST-GCN usually ~9 but since we use TGT structure we need to
    #       decrese the value so 3 or 5 maybe
    # Ks - in CoSign paper they say that 'distance partition strategy (k = 2, A0 = I, A1 = A)'
    #       i think this is it
    # act_funct - CoSign used reLu, later we can experiment with glu/gtu implemented in
    #       models.stgcn.layers; glu/gtu double the number of parameters in the time convs
    #       so using them may help with longer sequences but will increase the size of the model
    # droprate - 0.2 because why not, need to experiment with that

    def _create_args(self, gso_matrix):
        return STGCNArgs(
            Kt=3,
            Ks=CONST_KS,
            act_func="relu",
            graph_conv_type="graph_conv",
            gso=gso_matrix,
            enable_bias=True,
            droprate=0.2,
        )
