import os
from collections import defaultdict

import torch
import yaml
from torch.utils.data import DataLoader

from model import GlossClassifier
from pjm_dataloader import PJMDataset, build_gloss_vocab, collate_fn


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

    ckpt_path = os.path.join(config["system"]["checkpoint_dir"], "latest.pth")
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    else:
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

    ANN_DIR = config["data"]["annotation_dir"]
    ANN_TRAIN = config["data"]["train_ann"]
    ANN_DEV = config["data"]["dev_ann"]

    gloss2id = build_gloss_vocab(ANN_DIR, [ANN_TRAIN, ANN_DEV])
    id2gloss = {v: k for k, v in gloss2id.items()}

    dataset = PJMDataset(
        config["data"]["data_dir"],
        ANN_DIR,
        ANN_DEV,
        gloss2id,
    )

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

    with torch.no_grad():
        for frames, labels, lengths in dataloader:
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

    print(f"Overall: {correct}/{total} = {correct / total:.2%}\n")
    print(f"{'Gloss':<20} {'Correct':>8} {'Total':>8} {'Rate':>8}")
    print("-" * 48)
    for gid in sorted(per_gloss_total.keys(), key=lambda g: per_gloss_correct[g] / max(per_gloss_total[g], 1)):
        c = per_gloss_correct[gid]
        t = per_gloss_total[gid]
        name = id2gloss[gid]
        print(f"{name:<20} {c:>8} {t:>8} {c / t:>7.1%}")


if __name__ == "__main__":
    main()
