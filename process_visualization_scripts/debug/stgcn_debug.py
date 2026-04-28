"""This module contains debug code for our implementation of 1-stream CoSign architecture."""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from stgcn.gso import GSOGenerator
from stgcn.stgcn_src.models import STGCNGraphConv as STGCN
from process_visualization_scripts.debug_exporter import export_tensor_for_pil

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
    """Class that holds specific STGCN configuration arguments."""
    def __init__(self, Kt, Ks, act_func, graph_conv_type, gso, enable_bias, droprate):
        self.Kt = Kt
        self.Ks = Ks
        self.act_func = act_func
        self.graph_conv_type = graph_conv_type
        self.gso = gso
        self.enable_bias = enable_bias
        self.droprate = droprate


class STGCNCoSign1s(nn.Module):
    """CoSign-1s' ST-GCN debug implementation."""

    def __init__(self, config_path=None):
        super().__init__()

        # Fixing config path to point to original stgcn module
        if config_path is None:
            current_dir = Path(__file__).resolve().parent.parent / "stgcn"
            config_path = current_dir / "config.json"

        self.gso_generator = GSOGenerator(config_path)
        self.config = self.gso_generator.config

        self.gcn_out_dim = COSIGN_BLOCKS[-1][-1]
        self.fusion_out_dim = 1024

        self.offsets = {
            "body": 0,
            "face": 33,
            "mouth": 33,
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
                "hands": STGCN(
                    args=self._create_args(self.gso_generator.gsos["l_hand"]),
                    blocks=COSIGN_BLOCKS,
                    n_vertex=len(self.config["l_hand"]),
                ),
            }
        )

        num_groups_for_fusion = 5
        self.fusion_in_dim = num_groups_for_fusion * self.gcn_out_dim

        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(self.fusion_in_dim, self.fusion_out_dim, kernel_size=1),
            nn.GroupNorm(32, self.fusion_out_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

    def forward(self, x, keep_prob=0.8):
        if not self.training:
            keep_prob = 1.0

        x = _normalize_by_shoulder_width(self, x)

        centralized_groups = {}
        for name in ["body", "face", "mouth", "l_hand", "r_hand"]:
            local_indices = np.array(self.config[name])
            global_indices = local_indices + self.offsets[name]

            group_data = x[:, :, :, global_indices]

            root_point = group_data[:, :, :, 0].unsqueeze(-1)
            centralized_groups[name] = group_data - root_point

        # ====================================================
        # CHECKPOINT 2: Export anchored skeleton
        # Reconstruct the skeleton with each group locked to (0,0)
        # ====================================================
        if "checkpoint_2_anchored" not in getattr(self, "_exported", []):
            try:
                # Rebuild full tensor [B, 3, T, 553] with zeros
                B, C, T, _ = x.shape
                anchored_x = torch.zeros((B, C, T, 553), device=x.device, dtype=x.dtype)
                
                # We won't map 'mouth' since our visualizer uses full 'face'
                for name in ["body", "face", "l_hand", "r_hand"]:
                    local_indices = np.array(self.config[name])
                    global_indices = local_indices + self.offsets[name]
                    anchored_x[:, :, :, global_indices] = centralized_groups[name]
                
                export_tensor_for_pil(anchored_x, "process_visualization_scripts/checkpoint_2_anchored.npy")
                # Ensure we only export once per session
                if not hasattr(self, "_exported"):
                    self._exported = []
                self._exported.append("checkpoint_2_anchored")
            except Exception as e:
                print(f"[DEBUG] Failed to export anchored checkpoint: {e}")
        # ====================================================

        features = []
        for name, module_name in [
            ("body", "body"),
            ("face", "face"),
            ("mouth", "mouth"),
            ("l_hand", "hands"),
            ("r_hand", "hands"),
        ]:
            feat = self.gcn_modules[module_name](centralized_groups[name])
            feat = feat.mean(dim=-1)
            features.append(feat)

        v_groups = torch.stack(features, dim=1)

        phi = torch.bernoulli(
            torch.full(
                (v_groups.size(0), v_groups.size(1), 1, v_groups.size(3)),
                fill_value=keep_prob,
                device=v_groups.device,
                dtype=v_groups.dtype,
            )
        )

        phi_inv = 1.0 - phi
        v_masked = v_groups * phi
        v_masked_inv = v_groups * phi_inv

        v_masked = v_masked.flatten(1, 2)
        v_masked_inv = v_masked_inv.flatten(1, 2)

        v_fused_masked = self.fusion_mlp(v_masked)
        v_fused_masked_inv = self.fusion_mlp(v_masked_inv)

        v_fused = torch.stack([v_fused_masked, v_fused_masked_inv], dim=1)

        return v_fused

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