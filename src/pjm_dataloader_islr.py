"""
Data loading and preprocessing utilities for Isolated Sign Language Recognition (ISLR).

This module is responsible for loading `.npy` arrays containing raw MediaPipe skeleton landmarks,
filtering out irrelevant vertices, and converting them into uniform tensors of shape [T, V, C].
It also provides tools for dynamic vocabulary building based on segmented gloss annotations.
"""

import random

import numpy as np
import torch
import torch.nn.functional as F
from dataset_preprocess import (
    BLANK_GLOSS,
    END_OF_VIDEO,
    build_samples,
    load_gloss_map,
    map_gloss,
    unique_stems,
)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

POSE_LEN = 33
FACE_LEN = 478
LH_LEN = 21
RH_LEN = 21
TOTAL_V = POSE_LEN + FACE_LEN + LH_LEN + RH_LEN
LH_START = POSE_LEN + FACE_LEN
RH_START = POSE_LEN + FACE_LEN + LH_LEN


def _safe_part(raw: list, expected_len: int) -> np.ndarray:
    """
    Pads or cuts the landmark array to ensure it always has the exact expected length.

    Args:
        raw (list | np.ndarray): Raw landmark coordinates.
        expected_len (int): The required number of landmarks for this body part.

    Returns:
        np.ndarray: A padded or truncated numpy array of shape (expected_len, 4).
    """
    arr = np.array(raw, dtype=np.float32) if len(raw) > 0 else np.zeros((0, 4), dtype=np.float32)
    arr = arr.reshape(-1, 4)
    if arr.shape[0] == 0:
        return np.zeros((expected_len, 4), dtype=np.float32)
    if arr.shape[0] < expected_len:
        arr = np.pad(arr, ((0, expected_len - arr.shape[0]), (0, 0)))
    return arr[:expected_len]


def _convert_frame(frame_dict: dict) -> np.ndarray:
    """
    Combines pose, face, and hand landmarks from a single frame into one array (X, Y, confidence).

    Args:
        frame_dict (dict): Dictionary containing raw MediaPipe coordinates for the frame.

    Returns:
        np.ndarray: A concatenated array containing [X, Y, Confidence] for all chosen landmarks.
    """
    pose = _safe_part(frame_dict.get("pose", []), POSE_LEN)
    face = _safe_part(frame_dict.get("face", []), FACE_LEN)
    lh = _safe_part(frame_dict.get("lh", []), LH_LEN)
    rh = _safe_part(frame_dict.get("rh", []), RH_LEN)
    combined = np.concatenate([pose, face, lh, rh], axis=0)
    return combined[:, [0, 1, 3]]


def mirror_clip(clip: torch.Tensor) -> torch.Tensor:
    """
    Flips the skeleton horizontally to simulate left/right hand swapping for data augmentation.

    Args:
        clip (torch.Tensor): A tensor representing a sequence of frames, shape [Time, Vertices, Channels].

    Returns:
        torch.Tensor: The augmented (mirrored) sequence tensor.
    """
    flipped = clip.clone()
    flipped[:, LH_START:RH_START], flipped[:, RH_START:RH_START + RH_LEN] = (
        clip[:, RH_START:RH_START + RH_LEN].clone(),
        clip[:, LH_START:RH_START].clone(),
    )
    flipped[:, :POSE_LEN, 0] = 1.0 - flipped[:, :POSE_LEN, 0]
    flipped[:, LH_START:RH_START + RH_LEN, 0] = 1.0 - flipped[:, LH_START:RH_START + RH_LEN, 0]
    return flipped


def temporal_scale_clip(clip: torch.Tensor, scale_range: tuple = (0.8, 1.2), min_frames: int = 4) -> torch.Tensor:
    """
    Speeds up or slows down the video clip to make the model robust to different signing speeds.

    Args:
        clip (torch.Tensor): A tensor representing a sequence of frames, shape [Time, Vertices, Channels].
        scale_range (tuple[float, float]): Min and max scaling factor. Defaults to (0.8, 1.2).
        min_frames (int): Minimum length to maintain after scaling. Defaults to 4.

    Returns:
        torch.Tensor: The temporally stretched or compressed sequence tensor.
    """
    T, V, C = clip.shape
    scale = random.uniform(*scale_range)
    new_T = max(int(round(T * scale)), min_frames)
    if new_T == T:
        return clip
    x = clip.reshape(T, V * C).T.unsqueeze(0)
    x = F.interpolate(x, size=new_T, mode="linear", align_corners=False)
    return x.squeeze(0).T.reshape(new_T, V, C)


def build_gloss_vocab(annotation_dir: str, split_files: list, gloss_map_path: str = None, include_blank: bool = True) -> dict:
    """
    Reads annotation files and creates a dictionary mapping each unique sign to an integer ID.

    Args:
        annotation_dir (str): Path to the directory with annotation data.
        split_files (list[str]): List of paths defining the dataset splits.
        gloss_map_path (str, optional): Path to a custom mapping rules file. Defaults to None.
        include_blank (bool): Whether to reserve index 0 for the CTC blank token. Defaults to True.

    Returns:
        dict[str, int]: A dictionary mapping gloss string identifiers to their numeric IDs.
    """
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
    """
    PyTorch Dataset implementation for short, isolated sign clips.
    
    Data Processing Pipeline:
    1. Reads start/end frame indices from annotation files for a specific gloss.
    2. Loads the corresponding `.npy` skeleton sequence from disk (or RAM cache).
    3. Normalizes 543 MediaPipe landmarks down to a specific subset (Body, Face, Hands).
    4. Applies stochastic augmentations during training (Temporal Scaling, Horizontal Mirroring).
    5. Returns a padded tensor representation ready for the ST-GCN network.
    """

    def __init__(
        self,
        data_dir: str,
        annotation_dir: str,
        split_file: str,
        gloss2id: dict,
        gloss_map_path: str = None,
        add_blank_segments: bool = True,
        cache_videos: bool = True,
        warmup_cache: bool = False,
        mirror_prob: float = 0.0,
        temporal_scale: bool = False,
        temporal_scale_range: tuple = (0.8, 1.2),
    ):
        """
        Initializes the dataset and loads video paths and annotations.

        Args:
            data_dir (str): Path to directory with .npy sequences.
            annotation_dir (str): Path to directory with annotation files.
            split_file (str): Path to the list of dataset items.
            gloss2id (dict[str, int]): Dictionary mapping strings to class IDs.
            gloss_map_path (str, optional): Custom mapping rules. Defaults to None.
            add_blank_segments (bool): Include unlabeled sections as `<blank>`. Defaults to True.
            cache_videos (bool): Preload files into memory for speed. Defaults to True.
            warmup_cache (bool): Pre-fill cache upon initialization. Defaults to False.
            mirror_prob (float): Probability of horizontal flip. Defaults to 0.0.
            temporal_scale (bool): Apply temporal augmentation. Defaults to False.
            temporal_scale_range (tuple[float, float]): Range of scale multipliers. Defaults to (0.8, 1.2).
        """
        self.mirror_prob = mirror_prob
        self.temporal_scale = temporal_scale
        self.temporal_scale_range = temporal_scale_range
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
        self.blank_id = gloss2id.get(BLANK_GLOSS, -1)

        if warmup_cache and cache_videos:
            stems = unique_stems(self.samples)
            print(f"  warming cache for {len(stems)} unique videos...", flush=True)
            self.video_cache.warmup(stems)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_raw_video(self, stem: str):
        return self.video_cache.load(stem)

    def __getitem__(self, idx: int) -> tuple:
        """
        Gets a single video clip and its label, applying augmentations if enabled.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple[torch.Tensor, int, int]: A tuple containing:
                - The processed clip tensor of shape [Time, Vertices, Channels].
                - The integer ID of the target gloss.
                - The unpadded sequence length.
        """
        stem, start, end, gloss_id = self.samples[idx]

        raw = self._load_raw_video(stem)
        if raw is None:
            raise FileNotFoundError(f"No video file found for stem '{stem}' in {self.data_dir}")
        if end == END_OF_VIDEO:
            end = len(raw)
        frames = np.stack([_convert_frame(f) for f in raw[start:end]])
        clip = torch.tensor(frames, dtype=torch.float32)

        if self.temporal_scale and gloss_id != self.blank_id:
            clip = temporal_scale_clip(clip, self.temporal_scale_range)

        if gloss_id != self.blank_id and torch.rand(1).item() < self.mirror_prob:
            clip = mirror_clip(clip)

        return clip, gloss_id, clip.shape[0]


def collate_fn(batch: list) -> tuple:
    """
    Pads clips of different lengths with zeros so they can be grouped into a single batch.

    Args:
        batch (list[tuple]): A list of tuples containing (clip_tensor, gloss_id, length) for each sample.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - Padded clip sequences of shape [Batch, MaxTime, Vertices, Channels].
            - Target labels tensor of shape [Batch].
            - Sequence lengths tensor of shape [Batch].
    """
    clips = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    lengths = [item[2] for item in batch]

    padded = pad_sequence(clips, batch_first=True, padding_value=0.0)
    labels = torch.tensor(labels, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)

    return padded, labels, lengths