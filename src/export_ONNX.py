import torch

import model
import stgcn.stgcn
from model import CoSign1SModel

# Internal normalizations in ST-GCN can cause ONNX export issues so we bypass them
stgcn.stgcn._normalize_by_shoulder_width = lambda self, x, *args, **kwargs: x


def bypass_lstm(self, x, lengths=None):
    """
    Bypass dynamic sequence packing in LSTM for ONNX tracing.
    """
    x = x.transpose(1, 2)
    out, _ = self.lstm(x)
    return out


# Apply the monkey patch to the LSTM forward method
model.LSTM.forward = bypass_lstm

print("Loading the model...")
net = CoSign1SModel(num_classes=1000)
net.eval()

dummy_x = torch.zeros(1, 3, 60, 553)
dummy_lengths = torch.tensor([60])

print("Generating the ONNX file...")
torch.onnx.export(
    net, 
    (dummy_x, dummy_lengths), 
    "model_schema.onnx",
    input_names=["MediaPipe_Skeletons", "Sequence_Length"],
    output_names=["Predictions"]
)
print("ONNX export completed")