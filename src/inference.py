import os
from collections import Counter, defaultdict

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import GlossClassifier
from pjm_dataloader import PJMDataset, collate_fn


def load_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    device = torch.device(
        config["system"]["device"]
        if config["system"]["device"] != "auto"
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    ckpt_path = os.path.join(config["system"]["checkpoint_dir"], "latest.pth")
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    else:
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

    gloss2id = checkpoint["gloss2id"]
    id2gloss = {v: k for k, v in gloss2id.items()}
    print(f"Checkpoint vocab: {len(gloss2id)}")

    ANN_DIR = config["data"]["annotation_dir"]
    ANN_DEV = config["data"]["dev_ann"]

    dataset = PJMDataset(
        config["data"]["data_dir"],
        ANN_DIR,
        ANN_DEV,
        gloss2id,
        cache_videos=False,
        warmup_cache=False,
    )
    print(f"Dev samples: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = GlossClassifier(num_classes=len(gloss2id), dropout=0.0)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    correct = 0
    total = 0
    per_gloss_correct = defaultdict(int)
    per_gloss_total = defaultdict(int)
    per_gloss_mispred = defaultdict(Counter)

    with torch.no_grad():
        for frames, labels, lengths in tqdm(dataloader, desc="Inference"):
            frames = frames.permute(0, 3, 1, 2).to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            logits = model(frames, lengths)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for pred, label in zip(preds.tolist(), labels.tolist()):
                per_gloss_total[label] += 1
                if pred == label:
                    per_gloss_correct[label] += 1
                else:
                    per_gloss_mispred[label][pred] += 1

    print(f"Overall: {correct}/{total} = {correct / total:.2%}\n")
    print(f"{'Gloss':<20} {'Correct':>8} {'Total':>8} {'Rate':>8}  Top confusion")
    print("-" * 62)
    for gid in sorted(per_gloss_total.keys(), key=lambda g: per_gloss_correct[g] / max(per_gloss_total[g], 1)):
        c = per_gloss_correct[gid]
        t = per_gloss_total[gid]
        name = id2gloss[gid]
        top = per_gloss_mispred[gid].most_common(1)
        confused = id2gloss[top[0][0]] if top else "-"
        confused_n = top[0][1] if top else 0
        print(f"{name:<20} {c:>8} {t:>8} {c / t:>7.1%}  {confused:<15} ({confused_n}/{t})")


if __name__ == "__main__":
    main()
