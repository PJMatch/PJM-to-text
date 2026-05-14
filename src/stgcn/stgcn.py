"""This module contains code for our implementation of 1-stream CoSign architecture."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .gso import GSOGenerator
from .stgcn_src.models import STGCNGraphConv as STGCN

CONST_KS = 2

COSIGN_BLOCKS = [
    [3],  # input
    [64, 64, 64],  # layer 1
    [64, 64, 64],  # layer 2
    [64, 64, 64],  # layer 3
]


def _normalize_by_shoulder_width(self, x, left_idx=11, right_idx=12, eps=1e-6):
    """
    x: [B, C, T, V]
    Uses body pose shoulders to scale the whole skeleton so shoulder width ~= 1.
    """
    if x.size(-1) <= max(left_idx, right_idx):
        return x

    left = x[:, :2, :, left_idx]  # [B, 2, T]
    right = x[:, :2, :, right_idx]  # [B, 2, T]

    dist = torch.norm(left - right, dim=1)  # [B, T]
    valid = dist > eps

    scale = torch.ones(x.size(0), device=x.device, dtype=x.dtype)

    for b in range(x.size(0)):
        if valid[b].any():
            scale[b] = dist[b][valid[b]].median()

    scale = scale.view(-1, 1, 1, 1).clamp_min(eps)
    return x / scale


class STGCNArgs:
    """Class that holds specific STGCN configuration arguments.

    Later the object of this class is passed to the constructor of STGCN.
    """

    def __init__(self, Kt, Ks, act_func, graph_conv_type, gso, enable_bias, droprate):
        """STGCNArgs constructor."""
        self.Kt = Kt  # Temporal Kernel Size
        self.Ks = Ks  # Spatial Kernel Size
        self.act_func = act_func  # activation funciton, can be glu / gtu / relu / silu
        # can be 'cheb_graph_conv' or 'graph_conv'
        self.graph_conv_type = graph_conv_type
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

        self.fusion_out_dim = 1024

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

        # each per group (face + body + 2 * hand + mouth)
        num_groups_for_fusion = 5
        self.fusion_in_dim = num_groups_for_fusion * self.gcn_out_dim

        self.data_bn = nn.ModuleDict()
        for name in ["body", "face", "mouth", "l_hand", "r_hand"]:
            n_local = len(self.config[name])
            self.data_bn[name] = nn.BatchNorm1d(3 * n_local)

        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(self.fusion_in_dim, self.fusion_out_dim, kernel_size=1),
            nn.GroupNorm(32, self.fusion_out_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

    def forward(self, x, keep_prob=0.8):
        """Forward function of CoSign1s ST-GCN module.

        Args:
            x :[Batch, Channels, Timesteps, Vertices] -> [B, 2, 1024, T]
                x is original, all-point vector from the .npy files
            keep_prob: probability of keeping the feature during masking.

        Returns:
            v_fused : frame-wise feature for LSTM
        """
        if not self.training:
            keep_prob = 1.0

        x = _normalize_by_shoulder_width(self, x)

        centralized_groups = {}
        for name in ["body", "face", "mouth", "l_hand", "r_hand"]:
            local_indices = np.array(self.config[name])

            global_indices = local_indices + self.offsets[name]

            group_data = x[:, :, :, global_indices]  # [B, 3, T, V_local]

            root_point = group_data[:, :2, :, 0:1]  # [B, 2, T, 1]
            centralized = group_data.clone()
            # centralization - eqn. (2)
            centralized[:, :2] = group_data[:, :2] - root_point

            low_conf_mask = (group_data[:, 2:3] < 0.2).expand(-1, 2, -1, -1)
            centralized[:, :2] = torch.where(
                low_conf_mask,
                torch.zeros_like(centralized[:, :2]),
                centralized[:, :2],
            )

            B, C, T_, V_ = centralized.shape
            bn_in = centralized.permute(0, 1, 3, 2).reshape(B, C * V_, T_)
            bn_out = self.data_bn[name](bn_in)
            centralized = bn_out.reshape(B, C, V_, T_).permute(0, 1, 3, 2)

            centralized_groups[name] = centralized

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
            feat =             feat.mean(dim=-1)
            features.append(feat)

        v_groups = torch.stack(features, dim=1)

        t_B = v_groups.size(0)
        t_N = v_groups.size(1)
        t_T = v_groups.size(3)
        tau = 25
        num_chunks = int(np.ceil(t_T / tau))
        chunk_mask = torch.bernoulli(
            torch.full(
                (t_B, t_N, 1, num_chunks),
                fill_value=keep_prob,
                device=v_groups.device,
                dtype=v_groups.dtype,
            )
        )
        expanded_mask = chunk_mask.repeat_interleave(tau, dim=-1)
        phi = expanded_mask[..., :t_T]  # [B, 5, 1, T]

        phi_inv = 1.0 - phi
        v_masked = v_groups * phi  # [B, 5, 64, T]
        v_masked_inv = v_groups * phi_inv

        v_masked = v_masked.flatten(1, 2)  # [B, 320, T]
        v_masked_inv = v_masked_inv.flatten(1, 2)

        v_fused_masked = self.fusion_mlp(v_masked)  # [B, 1024, T]
        v_fused_masked_inv = self.fusion_mlp(v_masked_inv)

        v_fused = torch.stack([v_fused_masked, v_fused_masked_inv], dim=1)  # [B, 2, 1024, T]

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
            droprate=0.3,
        )
