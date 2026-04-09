import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


EPS = 1e-6


def load_phoenix_annotations(annotation_file: str):
    """
    seq_id|image_path|x|x|signer|GLOSS|language
    """
    ann = {}

    with open(annotation_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("|")
            if len(parts) < 7:
                continue

            seq_id = parts[0].strip()
            gloss_str = parts[5].strip()
            gloss_tokens = gloss_str.split() if gloss_str else []

            ann[seq_id] = {
                "gloss_tokens": gloss_tokens,
                "text": parts[6].strip(),
            }

    return ann


def build_gloss_vocab(annotation_files):
    glosses = set()

    for ann_file in annotation_files:
        ann = load_phoenix_annotations(ann_file)
        for item in ann.values():
            glosses.update(item["gloss_tokens"])

    gloss2id = {"<blank>": 0}
    for i, gloss in enumerate(sorted(glosses), start=1):
        gloss2id[gloss] = i

    id2gloss = {v: k for k, v in gloss2id.items()}
    return gloss2id, id2gloss


def random_temporal_scaling(frames, scale_range=(0.8, 1.2), min_frames=4):
    """
    frames: [T, V, C]
    Returns temporally resampled frames with linear interpolation.
    """
    scale_factor = random.uniform(*scale_range)

    T, V, C = frames.shape
    new_T = max(int(round(T * scale_factor)), min_frames)

    if new_T == T:
        return frames

    x = frames.reshape(T, V * C).transpose(0, 1).unsqueeze(0)  # [1, V*C, T]
    x = F.interpolate(x, size=new_T, mode="linear", align_corners=False)
    x = x.squeeze(0).transpose(0, 1).reshape(new_T, V, C)      # [new_T, V, C]

    return x


class PhoenixDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        annotation_file: str,
        gloss2id: dict,
        split: str = "train",
        use_temporal_aug: bool = True,
        temporal_scale_range=(0.8, 1.2),
    ):
        self.data_dir = Path(data_dir)
        self.file_paths = sorted(self.data_dir.glob("*.npy"))
        self.annotations = load_phoenix_annotations(annotation_file)
        self.gloss2id = gloss2id

        self.split = split
        self.use_temporal_aug = use_temporal_aug
        self.temporal_scale_range = temporal_scale_range

        self.samples = []
        for file_path in self.file_paths:
            seq_id = file_path.stem
            if seq_id not in self.annotations:
                continue

            gloss_tokens = self.annotations[seq_id]["gloss_tokens"]
            gloss_ids = [self.gloss2id[g]
                         for g in gloss_tokens if g in self.gloss2id]

            if len(gloss_ids) == 0:
                continue

            self.samples.append((file_path, seq_id, gloss_ids))

        if not self.samples:
            print(f"No matched samples found in {self.data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, seq_id, gloss_ids = self.samples[idx]

        sequence = np.load(file_path)  # expected [T, 1659]
        sequence = torch.tensor(sequence, dtype=torch.float32)
        sequence = sequence.reshape(sequence.shape[0], 553, 3)  # [T, V, C]

        if self.split == "train" and self.use_temporal_aug:
            sequence = random_temporal_scaling(
                sequence,
                scale_range=self.temporal_scale_range,
            )

        target = torch.tensor(gloss_ids, dtype=torch.long)

        return {
            "seq_id": seq_id,
            "frames": sequence,                 # [T, 553, 3]
            "target": target,                  # [S]
            "frame_len": sequence.shape[0],    # after temporal scaling
        }


def phoenix_ctc_collate_fn(batch):
    frames = [item["frames"] for item in batch]
    targets = [item["target"] for item in batch]
    seq_ids = [item["seq_id"] for item in batch]

    frame_lengths = torch.tensor([x.shape[0]
                                 for x in frames], dtype=torch.long)
    target_lengths = torch.tensor([y.shape[0]
                                  for y in targets], dtype=torch.long)

    padded_frames = pad_sequence(
        frames, batch_first=True, padding_value=0.0)   # [B, T, V, C]
    padded_targets = pad_sequence(
        targets, batch_first=True, padding_value=0)    # [B, S]

    return {
        "seq_ids": seq_ids,
        "frames": padded_frames,
        "frame_lengths": frame_lengths,
        "targets": padded_targets,
        "target_lengths": target_lengths,
    }
