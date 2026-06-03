"""Build file-level train/dev split files from per-gloss segmented annotations."""

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


def main():
    cfg = load_config()
    base = Path(__file__).parent
    ann_dir = cfg["annotation_dir"]
    train_out = base / "annotations" / "PJM_gloss.train.txt"
    dev_out = base / "annotations" / "PJM_gloss.dev.txt"

    min_sentences = cfg["min_sentences"]
    dev_ratio = 1.0 - cfg["train_ratio"]
    seed = cfg.get("seed", 42)
    random.seed(seed)

    gloss_sids = defaultdict(set)
    sid_stems = defaultdict(set)
    stem_sid = {}
    stem_segments = defaultdict(list)

    for fname in os.listdir(ann_dir):
        with open(os.path.join(ann_dir, fname)) as f:
            data = json.load(f)

        glosses = data.get("glosses", [])
        if not glosses or not isinstance(glosses[0], list):
            continue
        if not any(g == "EoR" for g, _ in glosses):
            continue

        parts = fname.replace(".json", "").split("_")
        sid = parts[1]
        stem = fname.replace(".json", "")
        stem_sid[stem] = sid
        sid_stems[sid].add(stem)

        for gloss, start in glosses:
            if gloss == "EoR":
                continue
            gloss_sids[gloss].add(sid)
            stem_segments[stem].append((gloss, start))

    dev_sids = set()
    splittable = {g for g, sids in gloss_sids.items() if len(sids) >= min_sentences}

    for gloss in sorted(splittable):
        sids = sorted(gloss_sids[gloss])
        random.shuffle(sids)
        n_dev = max(1, round(len(sids) * dev_ratio))
        for sid in sids[:n_dev]:
            dev_sids.add(sid)

    for gloss, sids in gloss_sids.items():
        if sids.issubset(dev_sids):
            moved = random.choice(sorted(sids))
            dev_sids.discard(moved)

    train_files = []
    dev_files = []
    train_seg = 0
    dev_seg = 0
    dev_glosses = set()
    train_glosses = set()

    for stem in sorted(stem_segments.keys()):
        sid = stem_sid[stem]
        if sid in dev_sids:
            dev_files.append(f"{stem}.npy")
            dev_seg += len(stem_segments[stem])
            for g, _ in stem_segments[stem]:
                dev_glosses.add(g)
        else:
            train_files.append(f"{stem}.npy")
            train_seg += len(stem_segments[stem])
            for g, _ in stem_segments[stem]:
                train_glosses.add(g)

    os.makedirs(train_out.parent, exist_ok=True)

    with open(train_out, "w") as f:
        f.write("\n".join(train_files) + "\n")

    with open(dev_out, "w") as f:
        f.write("\n".join(dev_files) + "\n")

    total_files = len(train_files) + len(dev_files)
    print(f"Files:    {len(train_files)} train / {len(dev_files)} dev ({len(dev_files)/total_files*100:.1f}%)")
    print(f"Segments: {train_seg} train / {dev_seg} dev ({dev_seg/(train_seg+dev_seg)*100:.1f}%)")
    print(f"Glosses:  {len(gloss_sids)} total / {len(splittable)} splittable / {len(dev_glosses & train_glosses)} in both")
    dev_only = dev_glosses - train_glosses
    if dev_only:
        print(f"Dev-only glosses (bad): {list(dev_only)}")


if __name__ == "__main__":
    main()
