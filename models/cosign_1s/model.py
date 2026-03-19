import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CoSignTemporalCNN(nn.Module):
    """Temporal module: C3-P2-C3-P2."""

    def __init__(self, in_dim=1024, hidden_dim=1024, dropout=0.2):
        super().__init__()

        self.conv1 = nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _pool_out_lengths(lengths, kernel_size=2, stride=2, padding=0, dilation=1):
        return torch.div(
            lengths + 2 * padding - dilation * (kernel_size - 1) - 1,
            stride,
            rounding_mode="floor",
        ) + 1

    def forward(self, x, lengths=None):
        # x: [B, C, T]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.pool2(x)

        if lengths is not None:
            lengths = self._pool_out_lengths(lengths)
            lengths = self._pool_out_lengths(lengths)

        return x, lengths



class LSTM(nn.Module):
    """Two-layer bidirectional LSTM contextual module."""

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
        # x: [B, C, T] -> [B, T, C]
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
    def __init__(self, feat_dim, vocab_size):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, feat_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # x: [B, T, C]
        x = F.normalize(x, dim=-1)
        w = F.normalize(self.weight, dim=-1)
        return torch.matmul(x, w.t())  # [B, T, vocab_size]
