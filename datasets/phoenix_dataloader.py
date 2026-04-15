"""Dataloader module for the Phoenix sign language dataset."""

from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

# PyTorch Dataset for loading PHOENIX-2014T feature sequences.


class PhoenixDataset(Dataset):
    """Dataset loader for sign language keypoint sequences."""

    def __init__(self, data_dir: str):
        """Initializes the PhoenixDataset."""
        self.data_dir = Path(data_dir)
        self.file_paths = list(self.data_dir.glob("*.npy"))

        if not self.file_paths:
            print(f"No .npy files found in {self.data_dir}")

    def __len__(self):
        """Returns the total number of sequences in the dataset."""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """Loads and formats a single sequence from the dataset by index."""
        file_path = self.file_paths[idx]
        sequence = np.load(file_path)  # (T, 1659)
        sequence = torch.tensor(sequence, dtype=torch.float32)
        sequence = sequence.reshape(sequence.shape[0], 553, 3)  # (T, 553, 3)
        return sequence


# Function to pad variable-length sequences in a batch.


def sequence_collate_fn(batch):
    """Pads sequences with zeros to match the longest sequence in the batch."""
    # Recording the original length of sequencess
    lengths = torch.tensor([seq.shape[0] for seq in batch], dtype=torch.long)

    # Pading sequences with zeros to match the longest sequence in the batch
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=0.0)
    # batch_first=True yields a tensor of shape (Batch_Size, Max_Seq_Length, 1659)

    return padded_batch, lengths
