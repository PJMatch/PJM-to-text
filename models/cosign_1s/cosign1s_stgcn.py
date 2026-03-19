"""This module contains code for our implementation of 1-stream CoSign architecture."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from models.cosign_1s.gso import GSOGenerator
from models.stgcn.models import STGCNGraphConv as STGCN

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

    def __init__(self, config_path=None):
        """CoSign1s ST-GCN's constructor."""
        super().__init__()

        if config_path is None:
            current_dir = Path(__file__).resolve().parent
            config_path = current_dir / "config.json"

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
                # both hands share the same weights and the same topology in config
                "hands": STGCN(
                    args=self._create_args(self.gso_generator.gsos["l_hand"]),
                    blocks=COSIGN_BLOCKS,
                    n_vertex=len(self.config["l_hand"]),
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

        for name, module_name in [
            ("body", "body"),
            ("face", "face"),
            ("mouth", "mouth"),
            ("l_hand", "hands"),
            ("r_hand", "hands"),
        ]:
            # gcn outputs [B, 64, T, V]
            feat = self.gcn_modules[module_name](centralized_groups[name])

            # global average pooling over vertecies (dim=-1)
            # result [B, 64, T]
            feat = feat.mean(dim=-1)
            features.append(feat)

        # print(f"DEBUG: Shape of one feature: {features[0].shape}")
        v = torch.cat(features, dim=1)  # [B, 320, T]

        # TODO: if we implement dropout mask it will be here
        # v = v * xi

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
