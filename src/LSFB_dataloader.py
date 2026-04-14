import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class CTCSentenceVocabulary:
    def __init__(self, json_path: str) -> None:
        self.word2id: Dict[str, int] = {"<blank>": 0, "<UNK>": 1}
        self.id2word: Dict[int, str] = {0: "<blank>", 1: "<UNK>"}
        self._build_vocab(json_path)

    def tokenize(self, text: str) -> List[str]:
        text = text.upper()
        text = re.sub(r"[^A-ZÀ-Ÿ0-9\s\']", "", text)
        return text.split()

    def _build_vocab(self, json_path: str) -> None:
        with open(json_path, "r", encoding="utf-8") as file_handler:
            all_annotations = json.load(file_handler)

        unique_words = set()
        for annotations in all_annotations.values():
            for annotation in annotations:
                words = self.tokenize(annotation["value"])
                unique_words.update(words)

        for word in sorted(unique_words):
            if word not in self.word2id:
                idx = len(self.word2id)
                self.word2id[word] = idx
                self.id2word[idx] = word

    def encode(self, text: str) -> List[int]:
        words = self.tokenize(text)
        return [self.word2id.get(w, 1) for w in words]

    def __len__(self) -> int:
        return len(self.word2id)


class SignLanguageCTCDataset(Dataset):
    def __init__(
        self,
        npy_path: str,
        json_path: str,
        vocab: CTCSentenceVocabulary,
        fps: int = 30,
        filter_mode: str = "lips_brows",
    ) -> None:
        self.data = np.load(npy_path)
        self.fps = fps
        self.vocab = vocab

        video_id = os.path.splitext(os.path.basename(npy_path))[0]

        with open(json_path, "r", encoding="utf-8") as file_handler:
            all_annotations = json.load(file_handler)
            if video_id not in all_annotations:
                raise ValueError(f"Missing key: {video_id}")
            self.annotations = all_annotations[video_id]

        base_points = list(range(75))

        if filter_mode == "jaw":
            jaw_idx = [
                10,
                21,
                54,
                58,
                67,
                93,
                103,
                109,
                127,
                132,
                136,
                148,
                149,
                150,
                152,
                162,
                172,
                176,
                234,
                251,
                284,
                288,
                297,
                323,
                332,
                338,
                356,
                361,
                365,
                377,
                378,
                379,
                389,
                397,
                400,
                454,
            ]
            face_points = [idx + 75 for idx in jaw_idx]
        elif filter_mode == "lips_brows":
            lips_brows_idx = [
                61,
                146,
                91,
                181,
                84,
                17,
                314,
                405,
                321,
                375,
                291,
                185,
                40,
                39,
                37,
                0,
                267,
                269,
                270,
                409,
                70,
                63,
                105,
                66,
                107,
                55,
                65,
                52,
                53,
                46,
                300,
                293,
                334,
                296,
                336,
                285,
                295,
                282,
                283,
                276,
            ]
            face_points = [idx + 75 for idx in lips_brows_idx]
        else:
            face_points = []

        selected_points = base_points + face_points
        self.flat_indices = []
        for point in selected_points:
            self.flat_indices.extend([point * 3, point * 3 + 1, point * 3 + 2])

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        annotation = self.annotations[idx]

        start_frame = int((annotation["start"] / 1000.0) * self.fps)
        end_frame = int((annotation["end"] / 1000.0) * self.fps)
        end_frame = min(end_frame, len(self.data))

        filtered_sequence = self.data[start_frame:end_frame, self.flat_indices]
        encoded_text = self.vocab.encode(annotation["value"])

        return (
            torch.tensor(filtered_sequence, dtype=torch.float32),
            torch.tensor(encoded_text, dtype=torch.long),
        )


def pad_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    video_seqs = [item[0] for item in batch]
    text_seqs = [item[1] for item in batch]

    video_lengths = torch.tensor([len(seq) for seq in video_seqs], dtype=torch.long)
    text_lengths = torch.tensor([len(seq) for seq in text_seqs], dtype=torch.long)

    padded_video = pad_sequence(video_seqs, batch_first=True, padding_value=0.0)
    padded_text = pad_sequence(text_seqs, batch_first=True, padding_value=0)

    return padded_video, padded_text, video_lengths, text_lengths
