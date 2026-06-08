import numpy as np
import torch
from dataset_preprocess import (
    BLANK_GLOSS,
    build_samples,
    load_gloss_map,
    map_gloss,
)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

POSE_LEN = 33
FACE_LEN = 478
LH_LEN = 21
RH_LEN = 21
TOTAL_V = POSE_LEN + FACE_LEN + LH_LEN + RH_LEN


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
        cache_videos=True,
    ):
        self.samples, self.video_cache = build_samples(
            data_dir=data_dir,
            annotation_dir=annotation_dir,
            split_file=split_file,
            gloss2id=gloss2id,
            gloss_map_path=gloss_map_path,
            add_blank_segments=add_blank_segments,
            cache_videos=cache_videos,
        )
        self.data_dir = self.video_cache.data_dir

    def __len__(self):
        return len(self.samples)

    def _load_raw_video(self, stem):
        return self.video_cache.load(stem)

    def __getitem__(self, idx):
        stem, start, end, gloss_id = self.samples[idx]

        raw = self._load_raw_video(stem)
        if raw is None:
            raise FileNotFoundError(f"No video file found for stem '{stem}' in {self.data_dir}")
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
