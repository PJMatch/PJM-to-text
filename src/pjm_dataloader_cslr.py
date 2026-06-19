"""
Data loading and sequence handling utilities for Continuous Sign Language Recognition (CSLR).

This module prepares long, unsegmented skeleton sequences representing full sentences.
It handles JSON annotation parsing, mapping sentence strings to sequences of numeric CTC tokens,
and padding variable-length clips into uniform batches for training.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

POSE_LEN = 33
FACE_LEN = 478
LH_LEN = 21
RH_LEN = 21
TOTAL_V = POSE_LEN + FACE_LEN + LH_LEN + RH_LEN


def load_pjm_annotations(annotation_dir: str, split_file: str) -> dict:
    """
    Loads the sequence of signs for each video from JSON annotation files.

    Args:
        annotation_dir (str): Path to the directory containing JSON annotation files.
        split_file (str): Path to the text file containing the list of video stems for the dataset split.

    Returns:
        dict[str, dict]: A dictionary mapping video stems to their corresponding 'gloss_tokens' and 'text'.
    """
    ann = {}
    annotation_dir = Path(annotation_dir)

    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            stem = line.strip().split(",")[0].replace(".npy", "")
            if not stem:
                continue

            json_path = annotation_dir / f"{stem}.json"
            if not json_path.exists():
                continue

            with open(json_path, "r", encoding="utf-8") as jf:
                data = json.load(jf)

            raw_glosses = data.get("glosses", [])
            if not raw_glosses:
                continue

            if isinstance(raw_glosses[0], list):
                gloss_tokens = [g for g, _ in raw_glosses if g != "EoR"]
            else:
                gloss_tokens = raw_glosses

            if not gloss_tokens:
                continue

            ann[stem] = {
                "gloss_tokens": gloss_tokens,
                "text": data.get("sentence", ""),
            }

    return ann


def build_gloss_vocab(annotation_dir: str, split_files: list) -> tuple:
    """
    Creates a dictionary mapping each sign in the dataset to a unique ID (with <blank> for CTC).

    Args:
        annotation_dir (str): Path to the directory containing JSON annotation files.
        split_files (list[str]): List of paths to split files (e.g., training and dev sets).

    Returns:
        tuple[dict[str, int], dict[int, str]]: A tuple containing:
            - gloss2id: Dictionary mapping gloss strings to integer IDs.
            - id2gloss: Dictionary mapping integer IDs back to gloss strings.
    """
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


def _safe_part(raw: list, expected_len: int) -> np.ndarray:
    """
    Pads or cuts the landmark array to a fixed size.

    Args:
        raw (list | np.ndarray): Raw landmark coordinates.
        expected_len (int): The required number of landmarks for this body part.

    Returns:
        np.ndarray: A standardized array of shape (expected_len, 4) of type float32.
    """
    arr = np.array(raw, dtype=np.float32) if len(raw) > 0 else np.zeros((0, 4), dtype=np.float32)
    arr = arr.reshape(-1, 4)
    if arr.shape[0] == 0:
        return np.zeros((expected_len, 4), dtype=np.float32)
    if arr.shape[0] < expected_len:
        arr = np.pad(arr, ((0, expected_len - arr.shape[0]), (0, 0)))
    return arr[:expected_len]


def _convert_pjm_frame(frame_dict: dict) -> np.ndarray:
    """
    Combines different body parts into a single frame representation with X, Y, and confidence.

    Args:
        frame_dict (dict): Dictionary containing raw MediaPipe landmarks for 'pose', 'face', 'lh', and 'rh'.

    Returns:
        np.ndarray: A concatenated array of landmarks for the frame containing [X, Y, Confidence].
    """
    pose = _safe_part(frame_dict.get("pose", []), POSE_LEN)
    face = _safe_part(frame_dict.get("face", []), FACE_LEN)
    lh = _safe_part(frame_dict.get("lh", []), LH_LEN)
    rh = _safe_part(frame_dict.get("rh", []), RH_LEN)

    combined = np.concatenate([pose, face, lh, rh], axis=0)
    return combined[:, [0, 1, 3]]


def _load_pjm_sequence(file_path: Path) -> torch.Tensor:
    """
    Loads a full video sequence from a numpy file and processes its frames.

    Args:
        file_path (Path | str): Path to the .npy file containing the raw MediaPipe sequence.

    Returns:
        torch.Tensor: A tensor of shape [Time, Vertices, Channels] containing the processed sequence.
    """
    raw = np.load(file_path, allow_pickle=True)
    frames = np.stack([_convert_pjm_frame(f) for f in raw])
    return torch.tensor(frames, dtype=torch.float32)


def random_temporal_scaling(frames: torch.Tensor, scale_range: tuple = (0.8, 1.2), min_frames: int = 4) -> torch.Tensor:
    """
    Changes the speed of the video randomly for data augmentation.

    Args:
        frames (torch.Tensor): The input skeleton sequence tensor of shape [Time, Vertices, Channels].
        scale_range (tuple[float, float]): The minimum and maximum scaling factors. Defaults to (0.8, 1.2).
        min_frames (int): The minimum allowed number of frames after scaling. Defaults to 4.

    Returns:
        torch.Tensor: The temporally scaled skeleton sequence.
    """
    scale_factor = random.uniform(*scale_range)
    T, V, C = frames.shape
    new_T = max(int(round(T * scale_factor)), min_frames)
    if new_T == T:
        return frames
    x = frames.reshape(T, V * C).transpose(0, 1).unsqueeze(0)
    x = F.interpolate(x, size=new_T, mode="linear", align_corners=False)
    x = x.squeeze(0).transpose(0, 1).reshape(new_T, V, C)
    return x


class PJMDataset(Dataset):
    """
    PyTorch Dataset for loading continuous sign language sequences.
    
    Data Processing Pipeline:
    1. Parses JSON annotation files to extract the ordered list of glosses for a given video.
    2. Maps each gloss to a numeric ID, reserving index 0 for the CTC `<blank>` token.
    3. Loads the full `.npy` sequence of MediaPipe frames and concatenates necessary body keypoints.
    4. Applies random temporal scaling to the entire sentence sequence to simulate varying signing speeds.
    5. Utilizes a custom collate function (`pjm_ctc_collate_fn`) to pad frames and targets.
    """
    
    def __init__(
        self,
        data_dir: str,
        annotation_dir: str,
        split_file: str,
        gloss2id: dict,
        split: str = "train",
        use_temporal_aug: bool = True,
        temporal_scale_range: tuple = (0.8, 1.2),
    ):
        """
        Initializes the dataset, loads annotations, and prepares sequence paths.

        Args:
            data_dir (str): Directory containing the extracted .npy sequence files.
            annotation_dir (str): Directory containing the JSON annotation files.
            split_file (str): Path to the split definition file.
            gloss2id (dict[str, int]): Dictionary mapping gloss tokens to numeric IDs.
            split (str): Dataset split type (e.g., "train", "dev", "test"). Defaults to "train".
            use_temporal_aug (bool): Whether to apply random temporal scaling. Defaults to True.
            temporal_scale_range (tuple[float, float]): Range for temporal augmentation. Defaults to (0.8, 1.2).
        """
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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
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


def pjm_ctc_collate_fn(batch: list) -> dict:
    """
    Pads frames and target labels so they can be processed together in a batch for CTC loss.

    Args:
        batch (list[dict]): A list of dictionaries containing single sample data 
            from the dataset (seq_id, frames, target, frame_len).

    Returns:
        dict[str, list | torch.Tensor]: A dictionary containing:
            - 'seq_ids' (list[str]): List of string identifiers for the sequences.
            - 'frames' (torch.Tensor): Padded skeleton tensor.
            - 'frame_lengths' (torch.Tensor): 1D tensor of original sequence lengths.
            - 'targets' (torch.Tensor): Padded target IDs tensor.
            - 'target_lengths' (torch.Tensor): 1D tensor of original target lengths.
    """
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