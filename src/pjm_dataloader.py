import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from phoenix_dataloader import random_temporal_scaling

POSE_LEN = 33
FACE_LEN = 478
LH_LEN = 21
RH_LEN = 21
TOTAL_V = POSE_LEN + FACE_LEN + LH_LEN + RH_LEN


def load_pjm_annotations(annotation_dir: str, split_file: str):
    ann = {}
    annotation_dir = Path(annotation_dir)

    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            stem = line.strip().replace(".npy", "")
            if not stem:
                continue

            json_path = annotation_dir / f"{stem}.json"
            if not json_path.exists():
                continue

            with open(json_path, "r", encoding="utf-8") as jf:
                data = json.load(jf)

            gloss_tokens = data.get("glosses", [])
            if not gloss_tokens:
                continue

            ann[stem] = {
                "gloss_tokens": gloss_tokens,
                "text": data.get("sentence", ""),
            }

    return ann


def build_gloss_vocab(annotation_dir, split_files):
    glosses = set()

    for split_file in split_files:
        ann = load_pjm_annotations(annotation_dir, split_file)
        for item in ann.values():
            glosses.update(item["gloss_tokens"])

    gloss2id = {"<blank>": 0}
    for i, gloss in enumerate(sorted(glosses), start=1):
        gloss2id[gloss] = i

    id2gloss = {v: k for k, v in gloss2id.items()}
    return gloss2id, id2gloss


def _safe_part(raw, expected_len):
    arr = np.array(raw, dtype=np.float32) if len(raw) > 0 else np.zeros((0, 4), dtype=np.float32)
    arr = arr.reshape(-1, 4)
    if arr.shape[0] == 0:
        return np.zeros((expected_len, 4), dtype=np.float32)
    if arr.shape[0] < expected_len:
        arr = np.pad(arr, ((0, expected_len - arr.shape[0]), (0, 0)))
    return arr[:expected_len]


def _convert_pjm_frame(frame_dict):
    pose = _safe_part(frame_dict.get("pose", []), POSE_LEN)
    face = _safe_part(frame_dict.get("face", []), FACE_LEN)
    lh = _safe_part(frame_dict.get("lh", []), LH_LEN)
    rh = _safe_part(frame_dict.get("rh", []), RH_LEN)

    combined = np.concatenate([pose, face, lh, rh], axis=0)
    # Output x, y, confidence (channels 0, 1, 3) instead of x, y, z (channels 0, 1, 2)
    return combined[:, [0, 1, 3]]


def _load_pjm_sequence(file_path):
    raw = np.load(file_path, allow_pickle=True)
    frames = np.stack([_convert_pjm_frame(f) for f in raw])
    return torch.tensor(frames, dtype=torch.float32)


class PJMDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        annotation_dir: str,
        split_file: str,
        gloss2id: dict,
        split: str = "train",
        use_temporal_aug: bool = True,
        temporal_scale_range=(0.8, 1.2),
    ):
        self.data_dir = Path(data_dir)
        self.annotations = load_pjm_annotations(annotation_dir, split_file)
        self.gloss2id = gloss2id

        self.split = split
        self.use_temporal_aug = use_temporal_aug
        self.temporal_scale_range = temporal_scale_range

        self.samples = []
        for stem, ann in self.annotations.items():
            signer = stem.split("_")[0]
            if signer == "8" or signer == "2":
                continue

            file_path = self.data_dir / f"{stem}.npy"
            if not file_path.exists():
                continue

            gloss_tokens = ann["gloss_tokens"]
            gloss_ids = [self.gloss2id[g] for g in gloss_tokens if g in self.gloss2id]

            if len(gloss_ids) == 0:
                continue

            self.samples.append((file_path, stem, gloss_ids))

        if not self.samples:
            print(f"No matched samples found in {self.data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, seq_id, gloss_ids = self.samples[idx]

        sequence = _load_pjm_sequence(file_path)

        if self.split == "train" and self.use_temporal_aug:
            sequence = random_temporal_scaling(
                sequence,
                scale_range=self.temporal_scale_range,
            )

        target = torch.tensor(gloss_ids, dtype=torch.long)

        return {
            "seq_id": seq_id,
            "frames": sequence,
            "target": target,
            "frame_len": sequence.shape[0],
        }


def pjm_ctc_collate_fn(batch):
    frames = [item["frames"] for item in batch]
    targets = [item["target"] for item in batch]
    seq_ids = [item["seq_id"] for item in batch]

    frame_lengths = torch.tensor([x.shape[0] for x in frames], dtype=torch.long)
    target_lengths = torch.tensor([y.shape[0] for y in targets], dtype=torch.long)

    padded_frames = pad_sequence(frames, batch_first=True, padding_value=0.0)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)

    return {
        "seq_ids": seq_ids,
        "frames": padded_frames,
        "frame_lengths": frame_lengths,
        "targets": padded_targets,
        "target_lengths": target_lengths,
    }
