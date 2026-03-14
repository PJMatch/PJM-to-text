import torch
import torch.nn as nn
from models.backbones import STGCN


class PJMTranslator(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.stgcn = STGCN(in_channels=3, graph_cfg=dict(layout="mediapipe"))
        self.lstm = nn.LSTM(256, 512, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(1024, num_classes + 1)

    def forward(self, x):
        # x: [Batch, Channels, Frames, Joints]
        features = self.backbone(x)
        # ... standard PyTorch logic from here ...
        return out
