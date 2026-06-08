import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from model import GlossClassifier
from pjm_dataloader import PJMDataset, build_gloss_vocab, collate_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reproduce_run", type=str, default=None)
    return parser.parse_args()


def load_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_training_run_manifest(manifest_path):
    with open(manifest_path) as f:
        return json.load(f)


def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def save_training_run(filepath, config, seed, deterministic, extra_info=None):
    run_info = {
        "config": config,
        "seed": seed,
        "deterministic": deterministic,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if extra_info:
        run_info.update(extra_info)
    with open(filepath, "w") as f:
        json.dump(run_info, f, indent=2)


def save_checkpoint(model, optimizer, epoch, gloss2id, val_acc, filepath):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "gloss2id": gloss2id,
        "val_acc": val_acc,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer, device):
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    print(f"Loaded '{filepath}' (epoch {epoch})")
    return epoch, model, optimizer


def try_resume_checkpoint(filepath, model, optimizer, device):
    """Resume training only when the checkpoint matches the current model."""
    try:
        return load_checkpoint(filepath, model, optimizer, device)
    except (KeyError, RuntimeError, ValueError) as exc:
        print(f"Skipping incompatible checkpoint '{filepath}'.")
        print(f"Reason: {exc}")
        print("Starting training from scratch.")
        return 0, model, optimizer


def evaluate(model, dataloader, criterion, device, id2gloss=None):
    model.eval()
    total_loss = 0.0
    total_top1 = 0
    total_top3 = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for frames, labels, lengths in dataloader:
            frames = frames.permute(0, 3, 1, 2).to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            logits = model(frames, lengths)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            top3 = logits.topk(3, dim=-1).indices
            total_top1 += (top3[:, 0] == labels).sum().item()
            total_top3 += (top3 == labels.unsqueeze(-1)).any(dim=-1).sum().item()
            total_samples += labels.size(0)

            all_preds.append(logits.argmax(dim=-1).cpu())
            all_labels.append(labels.cpu())

    avg_loss = total_loss / len(dataloader)
    top1 = total_top1 / max(total_samples, 1)
    top3 = total_top3 / max(total_samples, 1)

    preds = torch.cat(all_preds)
    targets = torch.cat(all_labels)

    n_classes = model.head.out_features
    conf = torch.zeros(n_classes, n_classes, dtype=torch.long)
    for t, p in zip(targets, preds):
        conf[t, p] += 1

    tp = conf.diag().float()
    fp = conf.sum(dim=0) - tp
    fn = conf.sum(dim=1) - tp

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1_per_class = 2 * precision * recall / (precision + recall + 1e-10)
    macro_f1 = f1_per_class.mean().item()

    worst = []
    if id2gloss is not None:
        for c in range(n_classes):
            total_c = (targets == c).sum().item()
            if total_c > 0:
                correct_c = (preds[targets == c] == c).sum().item()
                acc_c = correct_c / total_c
                if acc_c < 0.5:
                    name = id2gloss.get(c, f"id{c}")
                    worst.append((name, correct_c, total_c, acc_c))

    return avg_loss, top1, top3, macro_f1, worst, conf


def main():
    args = parse_args()

    if args.reproduce_run is not None:
        print(f"Loading run: {args.reproduce_run}")
        run_info = load_training_run_manifest(args.reproduce_run)
        config = run_info["config"]
        SEED = run_info["seed"]
        DETERMINISTIC = run_info["deterministic"]
    else:
        config = load_config()
        SEED = config["training"].get("seed", 42)
        DETERMINISTIC = config["training"].get("deterministic", True)

    EPOCHS = config["training"]["epochs"]
    BATCH_SIZE = config["training"]["batch_size"]
    LEARNING_RATE = float(config["training"]["learning_rate"])
    WEIGHT_DECAY = float(config["training"]["weight_decay"])
    MODEL_DROPOUT = config["model"]["dropout"]

    if config["system"]["device"] == "auto":
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        DEVICE = torch.device(config["system"]["device"])

    CHECKPOINT_DIR = config["system"]["checkpoint_dir"]
    DATA_DIR = config["data"]["data_dir"]
    ANN_DIR = config["data"]["annotation_dir"]
    ANN_TRAIN = config["data"]["train_ann"]
    ANN_DEV = config["data"]["dev_ann"]
    NUM_WORKERS = config["data"]["num_workers"]
    PIN_MEMORY = config["data"]["pin_memory"]

    OPTIMIZER_MILESTONES = config["optimizer"]["milestones"]
    OPTIMIZER_GAMMA = float(config["optimizer"]["gamma"])

    LOG_TENSORBOARD = config["logging"]["tensorboard"]
    LOG_WANDB = config["logging"]["wandb"] and WANDB_AVAILABLE
    LOG_DIR = config["logging"]["log_dir"]
    LOG_INTERVAL = config["logging"].get("log_interval", 50)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    set_seed(SEED, deterministic=DETERMINISTIC)

    run_id = f"isolated_{time.strftime('%Y%m%d_%H%M%S')}"
    tb_writer = SummaryWriter(os.path.join(LOG_DIR, run_id)) if LOG_TENSORBOARD else None
    if LOG_WANDB:
        wandb.init(project="gloss-sign-language", name=run_id, config=config, dir=LOG_DIR)

    print("Building vocabulary")
    gloss2id = build_gloss_vocab(ANN_DIR, [ANN_TRAIN, ANN_DEV])
    id2gloss = {v: k for k, v in gloss2id.items()}
    num_classes = len(gloss2id)
    print(f"Vocab size: {num_classes}")

    print("Loading datasets")
    train_dataset = PJMDataset(DATA_DIR, ANN_DIR, ANN_TRAIN, gloss2id)
    dev_dataset = PJMDataset(DATA_DIR, ANN_DIR, ANN_DEV, gloss2id)

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        worker_init_fn=seed_worker,
        generator=g,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        worker_init_fn=seed_worker,
    )

    print(f"Train: {len(train_dataset)} | Dev: {len(dev_dataset)} | Classes: {num_classes}")

    print(f"Initializing model on {DEVICE}")
    model = GlossClassifier(num_classes=num_classes, dropout=MODEL_DROPOUT)
    model.to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=OPTIMIZER_MILESTONES, gamma=OPTIMIZER_GAMMA
    )

    start_epoch = 0
    latest_ckpt = os.path.join(CHECKPOINT_DIR, "latest.pth")
    if os.path.exists(latest_ckpt):
        start_epoch, _, _ = try_resume_checkpoint(latest_ckpt, model, optimizer, DEVICE)

    best_acc = 0.0
    global_step = 0

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_train_loss = 0.0
        total_train_correct = 0
        total_train_samples = 0

        print(f"Epoch {epoch + 1:3d} | LR: {scheduler.get_last_lr()[0]:.2e}")

        for batch_idx, (frames, labels, lengths) in enumerate(train_loader):
            frames = frames.permute(0, 3, 1, 2).to(DEVICE)  #[B, C, T, V]
            labels = labels.to(DEVICE)
            lengths = lengths.to(DEVICE)

            optimizer.zero_grad()

            logits = model(frames, lengths)  #[B, V]
            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += loss.item()
            preds = logits.argmax(dim=-1)
            total_train_correct += (preds == labels).sum().item()
            total_train_samples += labels.size(0)
            global_step += 1

            if tb_writer and batch_idx % LOG_INTERVAL == 0:
                tb_writer.add_scalar("batch/loss", loss.item(), global_step)
                batch_acc = (preds == labels).float().mean().item()
                tb_writer.add_scalar("batch/acc", batch_acc, global_step)

            if batch_idx % 100 == 0:
                batch_acc = (preds == labels).float().mean().item()
                print(f"  Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {batch_acc:.3f}")

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_acc = total_train_correct / max(total_train_samples, 1)

        val_loss, val_top1, val_top3, val_mf1, val_worst, val_conf = evaluate(
            model, dev_loader, criterion, DEVICE, id2gloss
        )

        print(f"Epoch {epoch + 1} done. Train loss: {avg_train_loss:.4f} acc: {avg_train_acc:.3f} | "
              f"Val loss: {val_loss:.4f} top1: {val_top1:.3f} top3: {val_top3:.3f} mF1: {val_mf1:.3f}")

        if val_worst:
            print(f"  Glosses <50%: {len(val_worst)} — worst: {val_worst[0][0]} ({val_worst[0][3]*100:.0f}%)")

        epoch_metrics = {
            "val/loss": val_loss,
            "val/acc": val_top1,
            "val/top3": val_top3,
            "val/macro_f1": val_mf1,
            "train/loss": avg_train_loss,
            "train/acc": avg_train_acc,
            "train/lr": scheduler.get_last_lr()[0],
        }
        if tb_writer:
            for k, v in epoch_metrics.items():
                tb_writer.add_scalar(k, v, epoch + 1)

            n_classes = val_conf.shape[0]
            for c in range(min(5, len(val_worst))):
                name = val_worst[c][0]
                tb_writer.add_scalar(f"worst_glosses/{name}", val_worst[c][3], epoch + 1)

        if LOG_WANDB:
            wandb.log(epoch_metrics, step=epoch + 1)

        scheduler.step()

        save_checkpoint(model, optimizer, epoch + 1, gloss2id, val_top1, latest_ckpt)

        if val_top1 > best_acc:
            best_acc = val_top1
            best_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            save_checkpoint(model, optimizer, epoch + 1, gloss2id, val_top1, best_path)
            print(f"-> New best val acc: {best_acc:.3f}")

    if tb_writer:
        tb_writer.close()
    if LOG_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
