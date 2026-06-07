import torch
import torch.nn as nn
import torch.nn.functional as F
from stgcn.stgcn import STGCNCoSign1s


class AttentivePooling(nn.Module):
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
