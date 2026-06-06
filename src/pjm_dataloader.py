import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

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
    combined = np.concatenate([pose, face, lh, rh], axis=0)
    return combined[:, [0, 1, 3]]


def _load_segments(ann_dir, stem):
    """Return list of (gloss, start) and the EoR frame."""
    json_path = Path(ann_dir) / f"{stem}.json"
    with open(json_path) as f:
        data = json.load(f)
    entries = data.get("glosses", [])
    if not entries or not isinstance(entries[0], list):
        return [], None
    eor = None
    for g, s in entries:
        if g == "EoR":
            eor = s
            break
    return entries, eor


def build_gloss_vocab(annotation_dir, split_files):
    """Build gloss2id from segmented annotations."""
    ann_dir = Path(annotation_dir)
    glosses = set()

    for split_file in split_files:
        with open(split_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 3:
                    continue
                gloss_name = parts[2]
                glosses.add(gloss_name)

    gloss2id = {"<blank>": 0}
    for i, g in enumerate(sorted(glosses), start=1):
        gloss2id[g] = i

    return gloss2id


class PJMDataset(Dataset):
    """Dataset for isolated gloss clips."""

    def __init__(self, data_dir, annotation_dir, split_file, gloss2id):
        self.data_dir = Path(data_dir)
        self.samples = []  # (stem, start, end, gloss_id)

        stem_segments = defaultdict(list)
        with open(split_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 3:
                    continue
                stem, start_str, gloss = parts[0], parts[1], parts[2]
                stem = stem.replace(".npy", "")
                stem_segments[stem].append((int(start_str), gloss))

        for stem, segs in stem_segments.items():
            entries, eor = _load_segments(annotation_dir, stem)
            if not entries:
                continue

            seg_map = {}
            for i, (g, s) in enumerate(entries):
                if g == "EoR":
                    continue
                next_start = entries[i + 1][1] if i + 1 < len(entries) else eor
                seg_map[(g, s)] = next_start

            for start, gloss_name in segs:
                end = seg_map.get((gloss_name, start))
                if end is not None and end > start:
                    gid = gloss2id.get(gloss_name)
                    if gid is not None:
                        self.samples.append((stem, start, end, gid))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stem, start, end, gloss_id = self.samples[idx]

        npy_path = self.data_dir / f"{stem}.npy"
        raw = np.load(npy_path, allow_pickle=True)
        frames = np.stack([_convert_frame(f) for f in raw[start:end]])  #[T, 553, 3]
        clip = torch.tensor(frames, dtype=torch.float32)  #[T, 553, 3]

        return clip, gloss_id, clip.shape[0]


def collate_fn(batch):
    clips = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    lengths = [item[2] for item in batch]

    padded = pad_sequence(clips, batch_first=True, padding_value=0.0)  #[B, T_max, 553, 3]
    labels = torch.tensor(labels, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)

    return padded, labels, lengths
