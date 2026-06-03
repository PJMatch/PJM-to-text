"""Training loop for per-frame gloss classification."""

import argparse
import json
import os
import random
import time

import jiwer
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
    parser = argparse.ArgumentParser(description="Train gloss classification model")
    parser.add_argument(
        "--reproduce_run",
        type=str,
        default=None,
        help="Path to training_run.json file to reproduce a previous run",
    )
    return parser.parse_args()


def load_config(config_path="config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def load_training_run_manifest(manifest_path):
    with open(manifest_path) as f:
        run_info = json.load(f)
    return run_info


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


def save_checkpoint(model, optimizer, epoch, gloss2id, val_loss, seed, filepath):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "gloss2id": gloss2id,
        "val_loss": val_loss,
        "seed": seed,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer, device):
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    val_loss = checkpoint.get("val_loss", float("inf"))
    saved_seed = checkpoint.get("seed", "unknown")
    print(f"Loaded checkpoint '{filepath}' (epoch {epoch}, seed={saved_seed})")
    return epoch, model, optimizer


def downsample_labels(labels, frame_lengths, out_lengths):
    """labels: [B, T] padded with -100. Return [B, T_out] subsampled to match temporal CNN."""
    B = labels.size(0)
    T_out = out_lengths.max().item()
    downsampled = torch.full((B, T_out), -100, dtype=labels.dtype, device=labels.device)
    for b in range(B):
        in_len = frame_lengths[b].item()
        out_len = out_lengths[b].item()
        indices = (torch.arange(out_len, device=labels.device).float() * (in_len / out_len)).long()
        downsampled[b, :out_len] = labels[b, indices]
    return downsampled


def per_frame_accuracy(logits, labels):
    """Accuracy over labeled frames (ignore -100)."""
    preds = logits.argmax(dim=-1)  #[B, T]
    mask = labels != -100
    if mask.sum() == 0:
        return 0.0
    correct = (preds[mask] == labels[mask]).sum().item()
    return correct / mask.sum().item()


def collapse_predictions(logits, out_lengths):
    """Argmax and collapse consecutive duplicates into gloss sequences."""
    preds = logits.argmax(dim=-1)  #[B, T]
    sequences = []
    for b in range(preds.size(0)):
        seq = []
        prev = -1
        for t in range(out_lengths[b].item()):
            token = preds[b, t].item()
            if token != 0 and token != prev:
                seq.append(token)
            prev = token
        sequences.append(seq)
    return sequences


def evaluate(model, dataloader, criterion, id2gloss, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_labeled = 0
    all_hyps = []
    all_refs = []

    with torch.no_grad():
        for batch in dataloader:
            frames = batch["frames"].to(device).permute(0, 3, 1, 2)  #[B,C,T,V]
            frame_lengths = batch["frame_lengths"].to(device)
            labels = batch["labels"].to(device)  #[B,T] -100 padding

            logits, out_lengths = model(frames, frame_lengths)  #[B,T',V]

            d_labels = downsample_labels(labels, frame_lengths, out_lengths)  #[B,T']

            loss = criterion(logits.transpose(1, 2), d_labels)
            total_loss += loss.item()

            acc, labeled = per_frame_accuracy(logits, d_labels), (d_labels != -100).sum().item()
            total_acc += acc * labeled if labeled > 0 else 0
            total_labeled += labeled

            hyps = collapse_predictions(logits, out_lengths)
            for b in range(len(batch["seq_ids"])):
                ref_ids = labels[b][labels[b] != -100]
                ref_collapsed = []
                prev = -1
                for v in ref_ids:
                    token = v.item()
                    if token != 0 and token != prev:
                        ref_collapsed.append(token)
                    prev = token
                ref = [id2gloss.get(v, "") for v in ref_collapsed]
                hyp = [id2gloss.get(v, "") for v in hyps[b] if v != 0]
                all_refs.append(ref)
                all_hyps.append(hyp)

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / max(total_labeled, 1)

    ref_strs = [" ".join(r) if r else "<empty>" for r in all_refs]
    hyp_strs = [" ".join(h) if h else "<empty>" for h in all_hyps]
    wer = jiwer.wer(ref_strs, hyp_strs)

    return avg_loss, avg_acc, wer, all_hyps, all_refs


def main():
    args = parse_args()

    if args.reproduce_run is not None:
        print(f"Loading training run manifest from: {args.reproduce_run}")
        run_info = load_training_run_manifest(args.reproduce_run)
        config = run_info["config"]
        SEED = run_info["seed"]
        DETERMINISTIC = run_info["deterministic"]
        original_run_id = run_info.get("run_id", "unknown")
        REPRODUCING_RUN = True
        print(f"Reproducing run: {original_run_id}")
        print(f"Seed: {SEED} | Deterministic: {DETERMINISTIC}")
    else:
        config = load_config()
        SEED = config["training"].get("seed", 42)
        DETERMINISTIC = config["training"].get("deterministic", True)
        REPRODUCING_RUN = False
        original_run_id = None

    EPOCHS = config["training"]["epochs"]
    BATCH_SIZE = config["training"]["batch_size"]
    LEARNING_RATE = float(config["training"]["learning_rate"])
    WEIGHT_DECAY = float(config["training"]["weight_decay"])
    GRAD_CLIP = float(config["training"]["grad_clip"])
    INIT_WEIGHTS = config["training"].get("init_weights", None)
    MODEL_DROPOUT = config["model"]["dropout"]
    LSTM_HIDDEN = config["model"]["lstm_hidden"]
    FREEZE_BACKBONE = config["model"]["freeze_backbone"]
    INIT_BACKBONE = config["model"].get("init_backbone", None)

    if config["system"]["device"] == "auto":
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        DEVICE = torch.device(config["system"]["device"])

    CHECKPOINT_DIR = config["system"]["checkpoint_dir"]
    DATA_DIR_TRAIN = config["data"]["train_dir"]
    DATA_DIR_DEV = config["data"]["dev_dir"]
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
    HISTOGRAM_INTERVAL = config["logging"].get("histogram_interval", 5)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    set_seed(SEED, deterministic=DETERMINISTIC)

    if REPRODUCING_RUN:
        run_id = f"{original_run_id}_reproduction"
    else:
        run_id = f"gloss_{time.strftime('%Y%m%d_%H%M%S')}"

    tb_writer = SummaryWriter(os.path.join(LOG_DIR, run_id)) if LOG_TENSORBOARD else None
    if LOG_WANDB:
        wandb.init(project="gloss-sign-language", name=run_id, config=config, dir=LOG_DIR)

    print("Building vocabulary")
    gloss2id = build_gloss_vocab(ANN_DIR, [ANN_TRAIN, ANN_DEV])
    id2gloss = {v: k for k, v in gloss2id.items()}
    num_classes = len(gloss2id)
    print(f"Vocab size: {num_classes}")

    print("Loading datasets")
    train_dataset = PJMDataset(DATA_DIR_TRAIN, ANN_DIR, ANN_TRAIN, gloss2id)
    dev_dataset = PJMDataset(DATA_DIR_DEV, ANN_DIR, ANN_DEV, gloss2id, use_temporal_aug=False)

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
    model = GlossClassifier(
        num_classes=num_classes,
        dropout=MODEL_DROPOUT,
        lstm_hidden=LSTM_HIDDEN,
        freeze_backbone=FREEZE_BACKBONE,
    )
    model.to(DEVICE)

    if INIT_BACKBONE and os.path.exists(INIT_BACKBONE):
        model.load_pretrained(INIT_BACKBONE, DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    if INIT_WEIGHTS and os.path.exists(INIT_WEIGHTS):
        print(f"Loading checkpoint: {INIT_WEIGHTS}")
        ckpt = torch.load(INIT_WEIGHTS, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  -> Loaded from epoch {ckpt.get('epoch', '?')}")

    run_manifest_path = os.path.join(CHECKPOINT_DIR, f"{run_id}_training_run.json")
    extra_info = {
        "run_id": run_id,
        "num_classes": num_classes,
        "train_samples": len(train_dataset),
        "dev_samples": len(dev_dataset),
        "device": str(DEVICE),
    }
    if REPRODUCING_RUN:
        extra_info.update({
            "is_reproduction": True,
            "original_run_id": original_run_id,
            "original_manifest_path": args.reproduce_run,
        })
    save_training_run(run_manifest_path, config, SEED, DETERMINISTIC, extra_info=extra_info)

    start_epoch = 0
    latest_ckpt = os.path.join(CHECKPOINT_DIR, "latest.pth")
    if os.path.exists(latest_ckpt):
        start_epoch, _, _ = load_checkpoint(latest_ckpt, model, optimizer, DEVICE)

    best_wer = float("inf")
    global_step = 0

    for epoch in range(start_epoch, EPOCHS):
        current_lr = LEARNING_RATE
        if epoch >= OPTIMIZER_MILESTONES[0]:
            current_lr *= OPTIMIZER_GAMMA
        if len(OPTIMIZER_MILESTONES) > 1 and epoch >= OPTIMIZER_MILESTONES[1]:
            current_lr *= OPTIMIZER_GAMMA

        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        model.train()
        total_train_loss = 0.0
        total_train_acc = 0.0
        total_train_labeled = 0

        print(f"Epoch {epoch + 1:3d} | LR: {current_lr:.2e}")

        for batch_idx, batch in enumerate(train_loader):
            frames = batch["frames"].to(DEVICE).permute(0, 3, 1, 2)  #[B,C,T,V]
            frame_lengths = batch["frame_lengths"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)  #[B,T]

            optimizer.zero_grad()

            logits, out_lengths = model(frames, frame_lengths)  #[B,T',V]

            d_labels = downsample_labels(labels, frame_lengths, out_lengths)  #[B,T']

            loss = criterion(logits.transpose(1, 2), d_labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_train_loss += loss.item()
            acc, labeled = per_frame_accuracy(logits, d_labels), (d_labels != -100).sum().item()
            total_train_acc += acc * labeled if labeled > 0 else 0
            total_train_labeled += labeled
            global_step += 1

            if tb_writer and batch_idx % LOG_INTERVAL == 0:
                tb_writer.add_scalar("batch/loss", loss.item(), global_step)
                tb_writer.add_scalar("batch/acc", acc, global_step)

            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {acc:.3f}")

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_acc = total_train_acc / max(total_train_labeled, 1)

        val_loss, val_acc, val_wer, hyps, refs = evaluate(model, dev_loader, criterion, id2gloss, DEVICE)

        print(f"Epoch {epoch + 1} done. Train loss: {avg_train_loss:.4f} acc: {avg_train_acc:.3f} | "
              f"Val loss: {val_loss:.4f} acc: {val_acc:.3f} WER: {val_wer:.3f}")

        epoch_metrics = {
            "val/loss": val_loss,
            "val/acc": val_acc,
            "val/wer": val_wer,
            "train/loss": avg_train_loss,
            "train/acc": avg_train_acc,
            "train/lr": current_lr,
        }
        if tb_writer:
            for k, v in epoch_metrics.items():
                tb_writer.add_scalar(k, v, epoch + 1)

            for i in range(min(3, len(refs))):
                ref_str = " ".join(refs[i]) if refs[i] else "<empty>"
                hyp_str = " ".join(hyps[i]) if hyps[i] else "<empty>"
                tb_writer.add_text(
                    f"val/example_{i}",
                    f"**Ref:** {ref_str}\n\n**Hyp:** {hyp_str}",
                    epoch + 1,
                )

            if (epoch + 1) % HISTOGRAM_INTERVAL == 0 or epoch == 0:
                for name, param in model.named_parameters():
                    if param.ndim >= 2:
                        tb_writer.add_histogram(f"weights/{name}", param.data, epoch + 1)
                        if param.grad is not None:
                            tb_writer.add_histogram(f"grads/{name}", param.grad, epoch + 1)

        if LOG_WANDB:
            wandb.log(epoch_metrics, step=epoch + 1)

        save_checkpoint(model, optimizer, epoch + 1, gloss2id, val_loss, SEED, latest_ckpt)

        if val_wer < best_wer:
            best_wer = val_wer
            best_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            save_checkpoint(model, optimizer, epoch + 1, gloss2id, val_loss, SEED, best_path)
            print(f"-> New best WER: {best_wer:.3f}")

            extra_info.update({
                "best_wer": best_wer,
                "best_epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
            })
            save_training_run(run_manifest_path, config, SEED, DETERMINISTIC, extra_info=extra_info)

    if tb_writer:
        tb_writer.close()
    if LOG_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
