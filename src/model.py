import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class STGCNGroupEncoder(nn.Module):
    """
    Placeholder for a real group-specific ST-GCN encoder.
    """
    def __init__(self, num_nodes, in_channels, out_dim, dropout=0.2, name="group"):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.name = name

        self.placeholder = nn.Sequential(
            nn.Linear(num_nodes * in_channels, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        x: (B, T, V_group, C_in)
        returns: (B, T, D_out)
        """
        b, t, v, c = x.shape
        x = x.reshape(b, t, v * c)
        x = self.placeholder(x)
        return x


class FusionModule(nn.Module):
    """
    Fuses group-wise features into one frame-wise feature vector.
    """
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TemporalConvBlock(nn.Module):
    """
    Local temporal aggregation over frame-wise features.
    Input/Output: (B, T, D)
    """
    def __init__(self, dim, kernel_size=5, dropout=0.2):
        super().__init__()
        assert kernel_size % 2 == 1

        padding = kernel_size // 2

        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(dim),
            nn.Dropout(dropout),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, mask=None):
        residual = x
        x = x.transpose(1, 2)       # (B, D, T)
        x = self.net(x)
        x = x.transpose(1, 2)       # (B, T, D)
        x = self.relu(x + residual)

        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)

        return x


class ContextualModule(nn.Module):
    """
    CoSign-like contextual stage:
    1D temporal CNN -> UniLSTM
    """
    def __init__(
        self,
        feature_dim,
        temporal_layers=2,
        temporal_kernel=5,
        lstm_hidden=256,
        lstm_layers=1,
        dropout=0.2,
    ):
        super().__init__()

        self.temporal_blocks = nn.ModuleList([
            TemporalConvBlock(feature_dim, kernel_size=temporal_kernel, dropout=dropout)
            for _ in range(temporal_layers)
        ])

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

    def forward(self, x, lengths):
        """
        x: (B, T, D)
        lengths: (B,)
        returns:
            seq_features: (B, T_out, H)
            out_lengths: (B,)
        """
        max_len = x.size(1)
        device = x.device
        idx = torch.arange(max_len, device=device).unsqueeze(0)
        mask = idx < lengths.unsqueeze(1)

        x = x * mask.unsqueeze(-1).to(x.dtype)

        for block in self.temporal_blocks:
            x = block(x, mask=mask)

        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_out, _ = self.lstm(packed)

        x, out_lengths = pad_packed_sequence(
            packed_out,
            batch_first=True,
        )

        return x, out_lengths


class GlossHead(nn.Module):
    """
    Maps contextual features to gloss logits for CTC.
    """
    def __init__(self, in_dim, num_classes, dropout=0.2, blank_in_vocab=False):
        super().__init__()
        out_classes = num_classes if blank_in_vocab else num_classes + 1

        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_classes),
        )

    def forward(self, x):
        logits = self.net(x)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs


class OfflineCSLR(nn.Module):
    """
    CoSign-like offline baseline:
    grouped input -> group-specific encoders -> fusion ->
    contextual module -> gloss head

    Input:  (B, T, 553, 3)
    Output: log_probs (B, T_out, C), output_lengths (B,)
    """
    def __init__(
        self,
        num_classes,
        coord_dim=3,
        pose_nodes=33,
        face_nodes=478,
        left_hand_nodes=21,
        right_hand_nodes=21,
        pose_dim=64,
        face_dim=128,
        hand_dim=64,
        fused_dim=256,
        temporal_layers=2,
        temporal_kernel=5,
        lstm_hidden=256,
        lstm_layers=1,
        dropout=0.2,
        blank_in_vocab=False,
    ):
        super().__init__()

        self.coord_dim = coord_dim
        self.pose_nodes = pose_nodes
        self.face_nodes = face_nodes
        self.left_hand_nodes = left_hand_nodes
        self.right_hand_nodes = right_hand_nodes
        self.total_nodes = pose_nodes + face_nodes + left_hand_nodes + right_hand_nodes

        self.pose_encoder = STGCNGroupEncoder(
            num_nodes=pose_nodes,
            in_channels=coord_dim,
            out_dim=pose_dim,
            dropout=dropout,
            name="pose",
        )

        self.face_encoder = STGCNGroupEncoder(
            num_nodes=face_nodes,
            in_channels=coord_dim,
            out_dim=face_dim,
            dropout=dropout,
            name="face",
        )

        self.left_hand_encoder = STGCNGroupEncoder(
            num_nodes=left_hand_nodes,
            in_channels=coord_dim,
            out_dim=hand_dim,
            dropout=dropout,
            name="left_hand",
        )

        self.right_hand_encoder = STGCNGroupEncoder(
            num_nodes=right_hand_nodes,
            in_channels=coord_dim,
            out_dim=hand_dim,
            dropout=dropout,
            name="right_hand",
        )

        fusion_in_dim = pose_dim + face_dim + hand_dim + hand_dim
        self.fusion = FusionModule(
            in_dim=fusion_in_dim,
            out_dim=fused_dim,
            dropout=dropout,
        )

        self.contextual = ContextualModule(
            feature_dim=fused_dim,
            temporal_layers=temporal_layers,
            temporal_kernel=temporal_kernel,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dropout=dropout,
        )

        self.gloss_head = GlossHead(
            in_dim=lstm_hidden,
            num_classes=num_classes,
            dropout=dropout,
            blank_in_vocab=blank_in_vocab,
        )

    def split_groups(self, x):
        """
        x: (B, T, 553, 3)
        """
        pose_end = self.pose_nodes
        face_end = pose_end + self.face_nodes
        left_end = face_end + self.left_hand_nodes
        right_end = left_end + self.right_hand_nodes

        pose = x[:, :, :pose_end, :]
        face = x[:, :, pose_end:face_end, :]
        left_hand = x[:, :, face_end:left_end, :]
        right_hand = x[:, :, left_end:right_end, :]

        return pose, face, left_hand, right_hand

    def forward(self, x, lengths):
        """
        x: (B, T, 553, 3)
        lengths: (B,)
        returns:
            log_probs: (B, T_out, C)
            output_lengths: (B,)
        """
        if x.ndim != 4:
            raise ValueError(f"Expected input (B, T, V, C), got {tuple(x.shape)}")

        b, t, v, c = x.shape
        if v != self.total_nodes or c != self.coord_dim:
            raise ValueError(
                f"Expected (B, T, {self.total_nodes}, {self.coord_dim}), got {(b, t, v, c)}"
            )

        pose, face, left_hand, right_hand = self.split_groups(x)

        pose_feat = self.pose_encoder(pose)
        face_feat = self.face_encoder(face)
        left_hand_feat = self.left_hand_encoder(left_hand)
        right_hand_feat = self.right_hand_encoder(right_hand)

        fused = torch.cat(
            [pose_feat, face_feat, left_hand_feat, right_hand_feat],
            dim=-1,
        )

        frame_features = self.fusion(fused)

        contextual_features, output_lengths = self.contextual(
            frame_features,
            lengths,
        )

        log_probs = self.gloss_head(contextual_features)

        return log_probs, output_lengths
