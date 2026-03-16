"""This module contains code for our implementation of 1-stream CoSign architecture."""

import torch
import torch.nn as nn

from models.stgcn.models import STGCNGraphConv as STGCN


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


cs_blocks = [
    [3],  # input
    [64, 64, 64],  # layer 1
    [64, 64, 64],  # layer 2
    [64, 64, 64],  # layer 3
]


hand_gso_matrix = None
face_gso_matrix = None
body_gso_matrix = None
mouth_gso_matrix = None

# STGCNArgs parameter selection:
# Kt - in original ST-GCN usually ~9 but since we use TGT structure we need to
#       decrese the value so 3 or 5 maybe
# Ks - in CoSign paper they say that 'distance partition strategy (k = 2, A0 = I, A1 = A)'
#       i think this is it
# act_funct - CoSign used reLu, later we can experiment with glu/gtu implemented in
#       models.stgcn.layers; glu/gtu double the number of parameters in the time convs
#       so using them may help with longer sequences but will increase the size of the model
# droprate - 0.2 because why not, need to experiment with that

hands_args = STGCNArgs(
    Kt=3,
    Ks=2,
    act_func="relu",
    graph_conv_type="graph_conv",
    gso=hand_gso_matrix,
    enable_bias=True,
    droprate=0.2,
)

face_args = STGCNArgs(
    Kt=3,
    Ks=2,
    act_func="relu",
    graph_conv_type="graph_conv",
    gso=face_gso_matrix,
    enable_bias=True,
    droprate=0.2,
)

body_args = STGCNArgs(
    Kt=3,
    Ks=2,
    act_func="relu",
    graph_conv_type="graph_conv",
    gso=body_gso_matrix,
    enable_bias=True,
    droprate=0.2,
)

mouth_args = STGCNArgs(
    Kt=3,
    Ks=2,
    act_func="relu",
    graph_conv_type="graph_conv",
    gso=mouth_gso_matrix,
    enable_bias=True,
    droprate=0.2,
)


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

    def __init__(self):
        """CoSign1s ST-GCN's constructor."""
        super().__init__()

        self.gcn_out_dim = cs_blocks[-1][-1]
        # TODO: check the actual LSTM input size and put it here
        self.fusion_out_dim = 512  # this needs to be the size of LSTM input

        v_counts = {
            "face": 478,
            "mouth": 8,
            "body": 33,
            "hands": 21,
        }

        # TODO: Need to actually check project's indexing schema
        self.group_indices = {
            "body": list(range(0, 33)),
            "face": list(range(33, 511)),
            "l_hand": list(range(511, 532)),
            "r_hand": list(range(532, 553)),
            "mouth": [
                0,
                267,
                269,
                270,
                409,
                306,
                375,
                321,
                405,
                314,
                17,
                84,
                181,
                91,
                146,
                61,
                185,
                40,
                39,
                37,
            ],
        }

        self.gcn_modules = nn.ModuleDict(
            {
                "face": STGCN(args=face_args, blocks=cs_blocks, n_vertex=v_counts["face"]),
                "mouth": STGCN(args=mouth_args, blocks=cs_blocks, n_vertex=v_counts["mouth"]),
                "body": STGCN(args=body_args, blocks=cs_blocks, n_vertex=v_counts["body"]),
                # hands share the same weights
                "hands": STGCN(args=hands_args, blocks=cs_blocks, n_vertex=v_counts["hands"]),
            }
        )

        num_groups_for_fusion = 5  # each per group (face + body + 2 * hand + mouth)
        self.fusion_in_dim = num_groups_for_fusion * self.gcn_out_dim

        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.fusion_in_dim, self.fusion_out_dim),
            nn.BatchNorm1d(self.fusion_out_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )  # TODO: in CoSign paper they say smth about Bernouli distribution for dropout - eqn. (4)

    def forward(self, x):
        """Forward function of CoSign1s ST-GCN.

        Args:
            x :[Batch, Channels, Timesteps, Vertices] -> [B, 3, T, 553 (or less)]

        Returns:
            v_fused : # frame-wise feature for LSTM
        """
        groups = {
            "body": x[:, :, :, self.group_indices["body"]],
            "face": x[:, :, :, self.group_indices["face"]],
            "mouth": x[:, :, :, self.group_indices["face"]][:, :, :, self.group_indices["mouth"]],
            "l_hand": x[:, :, :, self.group_indices["l_hand"]],
            "r_hand": x[:, :, :, self.group_indices["r_hand"]],
        }

        roots = {
            "body": 11,  # left shoulder
            "face": 1,  # nose
            "mouth": 0,  # relative to all mouth idx
            "l_hand": 0,  # wrist
            "r_hand": 0,  # wrist
        }

        centralized_groups = {}
        for name, data in groups.items():
            # eqn. (2)
            root_idx = roots[name]
            root_point = data[:, :, :, root_idx].unsqueeze(-1)
            # J_tk = J_tk - J_t_r(g)
            centralized_groups[name] = data - root_point

        features = []

        features.append(self.gcn_modules["body"](centralized_groups["body"]))
        features.append(self.gcn_modules["face"](centralized_groups["face"]))
        features.append(self.gcn_modules["mouth"](centralized_groups["mouth"]))
        features.append(self.gcn_modules["hands"](centralized_groups["l_hand"]))
        features.append(self.gcn_modules["hands"](centralized_groups["r_hand"]))

        v = torch.cat(features, dim=-1)

        # TODO: if we implement dropout mask it will be here
        # v = v * xi

        # [B, T, 5*C] -> [B, 5*C, T] for BatchNorm1d in MLP
        v = v.permute(0, 2, 1)
        v_fused = self.fusion_mlp(v)

        return v_fused
