"""Build train/dev split files from per-gloss segmented annotations."""

import json
import os
import random
from collections import defaultdict
from pathlib import Path

import yaml


def load_config():
    """Read config.yaml and return the data section."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["data"]



def collect_segments(ann_dir):
    """Return dict: gloss -> list of (sentence_id, stem, start_frame)."""
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


def build_splits(gloss_segments, min_sentences, train_ratio, seed):
    """Split per gloss by sentence_id. Returns (train_lines, dev_lines)."""
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


def main():
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
