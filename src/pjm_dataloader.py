import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from phoenix_dataloader import random_temporal_scaling

POSE_LEN = 33
FACE_LEN = 478
LH_LEN = 21
RH_LEN = 21
TOTAL_V = POSE_LEN + FACE_LEN + LH_LEN + RH_LEN


def _safe_part(raw, expected_len):
    arr = np.array(raw, dtype=np.float32) if len(raw) > 0 else np.zeros((0, 4), dtype=np.float32)
    arr = arr.reshape(-1, 4)
    if arr.shape[0] == 0:
        return np.zeros((expected_len, 4), dtype=np.float32)
    if arr.shape[0] < expected_len:
        arr = np.pad(arr, ((0, expected_len - arr.shape[0]), (0, 0)))
    return arr[:expected_len]


def _convert_frame(frame_dict):
    """Convert one MediaPipe frame dict to (TOTAL_V, 3) tensor [x, y, confidence]."""
    pose = _safe_part(frame_dict.get("pose", []), POSE_LEN)
    face = _safe_part(frame_dict.get("face", []), FACE_LEN)
    lh = _safe_part(frame_dict.get("lh", []), LH_LEN)
    rh = _safe_part(frame_dict.get("rh", []), RH_LEN)
    combined = np.concatenate([pose, face, lh, rh], axis=0)  #[553,4]
    return combined[:, [0, 1, 3]]  #[553,3] x,y,confidence


def _load_sequence(file_path):
    """Load .npy file and return (T, 553, 3) tensor."""
    raw = np.load(file_path, allow_pickle=True)
    frames = np.stack([_convert_frame(f) for f in raw])
    return torch.tensor(frames, dtype=torch.float32)


def _load_gloss_segments(ann_dir, stem):
    """Return list of (gloss, start_frame) for a segmented file, EoR excluded."""
    json_path = Path(ann_dir) / f"{stem}.json"
    with open(json_path) as f:
        data = json.load(f)
    glosses = data.get("glosses", [])
    if not glosses or not isinstance(glosses[0], list):
        return []
    return [(g, s) for g, s in glosses if g != "EoR"]


def build_gloss_vocab(annotation_dir, split_files):
    """Build gloss2id from segmented annotations referenced in the split files."""
    ann_dir = Path(annotation_dir)
    glosses = set()

    for split_file in split_files:
        with open(split_file) as f:
            for line in f:
                stem = line.strip().replace(".npy", "")
                if not stem:
                    continue
                segments = _load_gloss_segments(ann_dir, stem)
                for g, _ in segments:
                    glosses.add(g)

    gloss2id = {"<blank>": 0}
    for i, g in enumerate(sorted(glosses), start=1):
        gloss2id[g] = i

    return gloss2id


class PJMDataset(Dataset):
    def __init__(
        self,
        data_dir,
        annotation_dir,
        split_file,
        gloss2id,
        use_temporal_aug=True,
        temporal_scale_range=(0.8, 1.2),
    ):
        self.data_dir = Path(data_dir)
        self.annotation_dir = annotation_dir
        self.gloss2id = gloss2id
        self.use_temporal_aug = use_temporal_aug
        self.temporal_scale_range = temporal_scale_range

        self.samples = []

        with open(split_file) as f:
            for line in f:
                stem = line.strip().replace(".npy", "")
                if not stem:
                    continue
                npy_path = self.data_dir / f"{stem}.npy"
                segments = _load_gloss_segments(annotation_dir, stem)
                self.samples.append((stem, npy_path, segments))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stem, npy_path, segments = self.samples[idx]

        frames = _load_sequence(npy_path)  #[T,553,3]
        original_len = frames.shape[0]
        total_frames = original_len

        if self.use_temporal_aug:
            frames = random_temporal_scaling(frames, scale_range=self.temporal_scale_range)
            total_frames = frames.shape[0]

        scale = total_frames / original_len

        labels = torch.full((total_frames,), -100, dtype=torch.long)
        for i in range(len(segments)):
            g, start = segments[i]
            end = segments[i + 1][1] if i + 1 < len(segments) else original_len
            scaled_start = round(start * scale)
            scaled_end = round(end * scale)
            if g in self.gloss2id:
                labels[scaled_start:scaled_end] = self.gloss2id[g]

        return {
            "seq_id": stem,
            "frames": frames,       #[T,553,3]
            "labels": labels,       #[T]  -100 for unlabeled/EoR frames
            "frame_len": total_frames,
        }


def collate_fn(batch):
    frames = [item["frames"] for item in batch]
    labels = [item["labels"] for item in batch]
    seq_ids = [item["seq_id"] for item in batch]

    frame_lengths = torch.tensor([x.shape[0] for x in frames], dtype=torch.long)

    padded_frames = pad_sequence(frames, batch_first=True, padding_value=0.0)  #[B,T_max,553,3]
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)   #[B,T_max]

    return {
        "seq_ids": seq_ids,
        "frames": padded_frames,
        "labels": padded_labels,
        "frame_lengths": frame_lengths,
    }
