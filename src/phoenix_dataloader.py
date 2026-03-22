import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import numpy as np


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


class PhoenixDataset(Dataset):
    def __init__(self, data_dir: str, annotation_file: str, gloss2id: dict):
        self.data_dir = Path(data_dir)
        self.file_paths = sorted(self.data_dir.glob("*.npy"))
        self.annotations = load_phoenix_annotations(annotation_file)
        self.gloss2id = gloss2id

        self.samples = []
        for file_path in self.file_paths:
            seq_id = file_path.stem
            if seq_id not in self.annotations:
                continue

            gloss_tokens = self.annotations[seq_id]["gloss_tokens"]
            gloss_ids = [self.gloss2id[g] for g in gloss_tokens if g in self.gloss2id]

            if len(gloss_ids) == 0:
                continue

            self.samples.append((file_path, seq_id, gloss_ids))

        if not self.samples:
            print(f"no matched samples found in {self.data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, seq_id, gloss_ids = self.samples[idx]

        sequence = np.load(file_path)  # (T, 1659)
        sequence = torch.tensor(sequence, dtype=torch.float32)
        sequence = sequence.reshape(sequence.shape[0], 553, 3)

        target = torch.tensor(gloss_ids, dtype=torch.long)

        sequence[:, :, 2] = 0.0 # remove Z axis

        return {
            "seq_id": seq_id,
            "frames": sequence,
            "target": target,
        }


def phoenix_ctc_collate_fn(batch):
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
