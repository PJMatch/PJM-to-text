"""
Build train/dev split files from per-gloss segmented annotations.

This script processes JSON annotation files to extract temporal segments for each gloss.
It then safely splits these segments into training and development sets based on unique 
sentence IDs to prevent data leakage between sets.
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path

import yaml


def load_config() -> dict:
    """
    Reads the config.yaml file and extracts the data section.

    Returns:
        dict: The configuration parameters related to data processing.
    """
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["data"]


def collect_segments(ann_dir: str) -> dict:
    """
    Scans annotation files and groups video segments by their corresponding gloss.

    Args:
        ann_dir (str): Path to the directory containing JSON annotation files.

    Returns:
        dict[str, list[tuple[str, str, int]]]: A dictionary mapping each gloss string 
            to a list of tuples containing (sentence_id, stem, start_frame).
    """
    gloss_segments = defaultdict(list)
    for fname in os.listdir(ann_dir):
        with open(os.path.join(ann_dir, fname)) as f:
            data = json.load(f)

        glosses = data.get("glosses", [])
        if not glosses or not isinstance(glosses[0], list):
            continue

        has_eor = any(g == "EoR" for g, _ in glosses)
        if not has_eor:
            continue

        parts = fname.replace(".json", "").split("_")
        sentence_id = parts[1]
        stem = fname.replace(".json", "")

        for gloss, start_frame in glosses:
            if gloss == "EoR":
                continue

            gloss_segments[gloss].append((sentence_id, stem, start_frame))

    return gloss_segments


def build_splits(gloss_segments: dict, min_sentences: int, train_ratio: float, seed: int) -> tuple:
    """
    Splits the collected segments into training and development sets based on sentence IDs.

    Ensures that the same base sentence doesn't appear in both training and dev sets.

    Args:
        gloss_segments (dict[str, list]): The dictionary of segments grouped by gloss.
        min_sentences (int): Minimum number of unique sentences required to perform a split. 
            If a gloss has fewer, all its instances go to the training set.
        train_ratio (float): The proportion of unique sentences to allocate to the training set.
        seed (int): Random seed for reproducible shuffling of sentence IDs.

    Returns:
        tuple[list[str], list[str], int, int, set[str]]: A tuple containing:
            - List of formatted string lines for the training split file.
            - List of formatted string lines for the development split file.
            - Total number of training segments.
            - Total number of development segments.
            - Set of unique glosses present in the development set.
    """
    random.seed(seed)
    train_lines = []
    dev_lines = []
    train_count = 0
    dev_count = 0
    dev_glosses = set()

    for gloss, segments in sorted(gloss_segments.items()):
        sid_to_segments = defaultdict(list)
        for sid, stem, start in segments:
            sid_to_segments[sid].append((stem, start))

        unique_sids = sorted(sid_to_segments.keys())

        if len(unique_sids) < min_sentences:
            for sid, segs in sid_to_segments.items():
                for stem, start in segs:
                    train_lines.append(f"{stem}.npy,{start},{gloss}")
                    train_count += 1
            continue

        shuffled = list(unique_sids)
        random.shuffle(shuffled)

        n_train = max(1, round(len(shuffled) * train_ratio))
        n_dev = len(shuffled) - n_train
        n_dev = max(1, n_dev)
        n_train = len(shuffled) - n_dev

        train_sids = set(shuffled[:n_train])
        dev_sids = set(shuffled[n_train:])

        for sid, segs in sid_to_segments.items():
            for stem, start in segs:
                if sid in train_sids:
                    train_lines.append(f"{stem}.npy,{start},{gloss}")
                    train_count += 1
                else:
                    dev_lines.append(f"{stem}.npy,{start},{gloss}")
                    dev_count += 1
                    dev_glosses.add(gloss)

    return train_lines, dev_lines, train_count, dev_count, dev_glosses


def main() -> None:
    """
    Main execution pipeline for building dataset splits.

    Loads the configuration, collects segments from raw annotations, applies the 
    split logic, and saves the resulting mappings to text files.

    Returns:
        None
    """
    cfg = load_config()
    base = Path(__file__).parent

    ann_dir = cfg["annotation_dir"] 
    train_path = base / "annotations" / "PJM_gloss.train.txt"
    dev_path = base / "annotations" / "PJM_gloss.dev.txt"

    gloss_segments = collect_segments(ann_dir)
    train_lines, dev_lines, train_count, dev_count, dev_glosses = build_splits(
        gloss_segments,
        cfg["min_sentences"],
        cfg["train_ratio"],
        seed=cfg.get("seed", 42),
    )

    os.makedirs(train_path.parent, exist_ok=True)

    with open(train_path, "w") as f:
        f.write("\n".join(train_lines) + "\n")

    with open(dev_path, "w") as f:
        f.write("\n".join(dev_lines) + "\n")

    total = train_count + dev_count
    print(f"Segments: {train_count} train / {dev_count} dev ({dev_count/total*100:.1f}% dev)")
    print(f"Glosses: {len(gloss_segments)} total, {len(dev_glosses)} in dev")
    print(f"Train file: {train_path}")
    print(f"Dev file:   {dev_path}")


if __name__ == "__main__":
    main()