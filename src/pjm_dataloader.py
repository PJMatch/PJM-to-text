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
EOR_TOKEN = "EoR"
BLANK_GLOSS = "blank"


def _default_gloss_map_path():
    return Path(__file__).resolve().parent / "annotations" / "gloss_map.json"


def load_gloss_map(gloss_map_path=None):
    """Load optional gloss canonicalization map without modifying dataset files."""
    gloss_map_path = Path(gloss_map_path) if gloss_map_path is not None else _default_gloss_map_path()
    if not gloss_map_path.exists():
        return {}

    with open(gloss_map_path, encoding="utf-8") as f:
        return json.load(f)


def map_gloss(gloss_name, gloss_map):
    """Return canonical gloss name using gloss_map, falling back to the original name."""
    return gloss_map.get(gloss_name, gloss_name)


def _safe_part(raw, expected_len):
    "Sanitize data from mediapipe to a constant shape of (expected_len, 4) [x, y, z, confidence]."
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
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("glosses", [])
    if not entries or not isinstance(entries[0], list):
        return [], None
    eor = None
    for g, s in entries:
        if g == EOR_TOKEN:
            eor = s
            break
    return entries, eor


def _load_video_length(data_dir, stem):
    npy_path = Path(data_dir) / f"{stem}.npy"
    if not npy_path.exists():
        return None

    raw = np.load(npy_path, allow_pickle=True)
    return len(raw)


def build_gloss_vocab(annotation_dir, split_files, gloss_map_path=None, include_blank=True):
    """Build gloss2id from segmented annotations."""
    glosses = set()
    gloss_map = load_gloss_map(gloss_map_path)

    for split_file in split_files:
        with open(split_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 3:
                    continue
                gloss_name = parts[2]
                glosses.add(map_gloss(gloss_name, gloss_map))

    gloss2id = {}
    if include_blank:
        gloss2id[BLANK_GLOSS] = 0

    start_idx = len(gloss2id)
    for i, g in enumerate(sorted(glosses), start=start_idx):
        if g == BLANK_GLOSS:
            continue
        gloss2id[g] = i

    return gloss2id


class PJMDataset(Dataset):
    """Dataset for isolated gloss clips."""

    def __init__(
        self,
        data_dir,
        annotation_dir,
        split_file,
        gloss2id,
        gloss_map_path=None,
        add_blank_segments=True,
    ):
        self.data_dir = Path(data_dir)
        self.samples = []  # (stem, start, end, gloss_id)
        self.gloss_map = load_gloss_map(gloss_map_path)
        self.add_blank_segments = add_blank_segments

        stem_segments = defaultdict(list)
        with open(split_file, encoding="utf-8") as f:
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

            regular_entries = [(g, s) for g, s in entries if g != EOR_TOKEN]

            if self.add_blank_segments:
                blank_gid = gloss2id.get(BLANK_GLOSS)
                if blank_gid is not None:
                    if regular_entries:
                        first_start = min(start for _, start in regular_entries)
                        if first_start > 0:
                            self.samples.append((stem, 0, first_start, blank_gid))
                    elif eor is not None and eor > 0:
                        self.samples.append((stem, 0, eor, blank_gid))

                    video_len = _load_video_length(self.data_dir, stem)
                    if eor is not None and video_len is not None and video_len > eor:
                        self.samples.append((stem, eor, video_len, blank_gid))

            seg_map = {}
            for i, (g, s) in enumerate(entries):
                if g == EOR_TOKEN:
                    continue
                next_start = entries[i + 1][1] if i + 1 < len(entries) else eor
                seg_map[(g, s)] = next_start

            for start, gloss_name in segs:
                end = seg_map.get((gloss_name, start))
                if end is not None and end > start:
                    mapped_gloss = map_gloss(gloss_name, self.gloss_map)
                    gid = gloss2id.get(mapped_gloss)
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
