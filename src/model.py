import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from stgcn.stgcn import STGCNCoSign1s


class CoSignTemporalCNN(nn.Module):
    """Temporal convolutional network to extract features over time using 1D CNNs (C3-P2-C3-P2)."""

    def __init__(self, in_dim=1024, hidden_dim=1024, dropout=0.2):
        super().__init__()

        self.conv1 = nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, hidden_dim)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, hidden_dim)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _pool_out_lengths(lengths, kernel_size=2, stride=2, padding=0, dilation=1):
        """Calculates the new sequence length after a max pooling operation."""
        out_lengths = (
            torch.div(
                lengths + 2 * padding - dilation * (kernel_size - 1) - 1,
                stride,
                rounding_mode="floor",
            )
            + 1
        )
        return torch.clamp(out_lengths, min=1)

    def forward(self, x, lengths=None):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.pool2(x)

        if lengths is not None:
            lengths = self._pool_out_lengths(lengths)
            lengths = self._pool_out_lengths(lengths)

        return x, lengths


class LSTM(nn.Module):
    """Bidirectional LSTM to capture sequential context from temporal features."""

    def __init__(self, input_dim=1024, hidden_size=512, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, x, lengths=None):
        x = x.transpose(1, 2)

        if lengths is None:
            out, _ = self.lstm(x)
            return out

        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        return out


class SharedGlossHead(nn.Module):
    """Classification head that uses cosine similarity instead of standard linear projection."""
    def __init__(self, feat_dim, vocab_size):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, feat_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = nn.Parameter(torch.tensor(25.0))

    def forward(self, x):
        x_norm = F.normalize(x, dim=-1)
        w_norm = F.normalize(self.weight, dim=-1)
        sim = torch.matmul(x_norm, w_norm.t())
        return sim * self.scale


class CoSign1SModel(nn.Module):
    """
    One-stream CoSign model with complementary masking:

    ST-GCN (with masking & fusion) -> [B, 2, 1024, T]
        branch 0: phi
        branch 1: 1 - phi

    For each branch:
        1D CNN -> aux gloss head (CTC)
        BiLSTM -> main gloss head (CTC)
    """

    def __init__(
        self,
        num_classes,
        stgcn_config_path=None,
        feat_dim=1024,
        lstm_hidden=512,
        dropout=0.2,
    ):
        super().__init__()

        self.STGCN = STGCNCoSign1s(config_path=stgcn_config_path)

        self.temporal_cnn = CoSignTemporalCNN(
            in_dim=feat_dim,
            hidden_dim=feat_dim,
            dropout=dropout,
        )

        self.context_lstm = LSTM(
            input_dim=feat_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            dropout=dropout,
        )

        self.gloss_head = SharedGlossHead(
            feat_dim=2 * lstm_hidden,
            vocab_size=num_classes,
        )

    def _forward_branch(self, x_branch, lengths):
        """
        One complementary branch:
        x_branch: [B, 1024, T]
        lengths:  [B]
        """
        cnn_feat, out_lengths = self.temporal_cnn(x_branch, lengths)

        B, C, T_prime = cnn_feat.shape
        device = cnn_feat.device

        time_steps = torch.arange(T_prime, device=device).unsqueeze(0)
        length_tensor = out_lengths.unsqueeze(1)

        mask = time_steps < length_tensor
        mask = mask.unsqueeze(1).expand_as(cnn_feat)

        cnn_feat = cnn_feat * mask

        aux_feat = cnn_feat.transpose(1, 2)
        aux_logits = self.gloss_head(aux_feat)

        lstm_out = self.context_lstm(cnn_feat, out_lengths)
        main_logits = self.gloss_head(lstm_out)

        return {
            "cnn_feat": cnn_feat,
            "aux_logits": aux_logits,
            "main_logits": main_logits,
            "logit_lengths": out_lengths,
        }

    def forward(self, x, lengths, keep_prob=1):
        branches = self.STGCN(x, keep_prob=keep_prob)  #[B, 2, 1024, T]
        branch_phi = branches[:, 0, ...]
        branch_phi_inv = branches[:, 1, ...]

        out_phi = self._forward_branch(branch_phi, lengths)
        out_phi_inv = self._forward_branch(branch_phi_inv, lengths)

        return {
            "phi": out_phi,
            "phi_inv": out_phi_inv,
        }


class AttentivePooling(nn.Module):
    """Collapses a sequence of frames into a single feature vector using learned attention weights."""
    def __init__(self, feat_dim):
        super().__init__()
        self.score = nn.Linear(feat_dim, 1)

    def forward(self, x, lengths):
        scores = self.score(x).squeeze(-1)  #[B, T]

        B, T = scores.shape
        mask = torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        scores[~mask] = float("-inf")

        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  #[B, T, 1]
        pooled = (x * weights).sum(dim=1)  #[B, D]
        return pooled


class GlossClassifier(nn.Module):
    """Isolated gloss classifier: ST-GCN -> attention pooling -> linear head."""

    def __init__(self, num_classes, dropout=0.2):
        super().__init__()

        self.STGCN = STGCNCoSign1s()
        self.pool = AttentivePooling(feat_dim=1024)
        self.head = nn.Linear(1024, num_classes)

    def forward(self, x, lengths):
        branches = self.STGCN(x, keep_prob=1.0)  #[B, 2, 1024, T]
        feat = branches[:, 0, :, :]  #[B, 1024, T]
        feat = feat.transpose(1, 2)  #[B, T, 1024]

        pooled = self.pool(feat, lengths)  #[B, 1024]
        logits = self.head(pooled)  #[B, V]
        return logits
