import argparse
import json
import os
import random
import time

import jiwer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from model import CoSign1SModel
from phoenix_dataloader import (
    PhoenixDataset,
    build_gloss_vocab as build_phoenix_vocab,
    phoenix_ctc_collate_fn,
)
from pjm_dataloader import PJMDataset, build_gloss_vocab as build_pjm_vocab, pjm_ctc_collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Train CoSign sign language recognition model")
    parser.add_argument(
        "--reproduce_run",
        type=str,
        default=None,
        help="Path to training_run.json file to reproduce a previous run",
    )
    return parser.parse_args()


def mirror_batch(frames, frame_lengths):
    mirrored = frames.clone()
    mirrored[:, :, :, 0] *= -1
    return mirrored


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_training_run_manifest(manifest_path):
    """Load a training_run.json manifest file."""
    with open(manifest_path, "r") as f:
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
    print(f"Torch version: {run_info.get('torch_version', 'unknown')}")
    if "best_wer" in run_info:
        print(
            f"Original best WER: {run_info['best_wer']:.3f} at epoch {run_info.get('best_wer_epoch', 'unknown')}"
        )
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
KL_WEIGHT = float(config["training"]["kl_weight"])
MIRROR_DATA = config["training"]["mirror_data"]

if config["system"]["device"] == "auto":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device(config["system"]["device"])

CHECKPOINT_DIR = config["system"]["checkpoint_dir"]
DATA_DIR_TRAIN = config["data"]["train_dir"]
DATA_DIR_DEV = config["data"]["dev_dir"]
DATA_DIR_TEST = config["data"]["test_dir"]
ANN_TRAIN = config["data"]["train_ann"]
ANN_DEV = config["data"]["dev_ann"]
ANN_TEST = config["data"]["test_ann"]
NUM_WORKERS = config["data"]["num_workers"]
PIN_MEMORY = config["data"]["pin_memory"]

MODEL_DROPOUT = config["model"]["dropout"]

LOG_TENSORBOARD = config["logging"]["tensorboard"]
LOG_WANDB = config["logging"]["wandb"] and WANDB_AVAILABLE
LOG_INTERVAL = config["logging"]["log_interval"]
HISTOGRAM_INTERVAL = config["logging"]["histogram_interval"]
GRADIENT_INTERVAL = config["logging"]["gradient_interval"]
LOG_DIR = config["logging"]["log_dir"]

OPTIMIZER_MILESTONES = config["optimizer"]["milestones"]
OPTIMIZER_GAMMA = float(config["optimizer"]["gamma"])


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


def load_checkpoint(filepath, model, optimizer):
    checkpoint = torch.load(filepath, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    val_loss = checkpoint.get("val_loss", float("inf"))
    saved_seed = checkpoint.get("seed", "unknown")
    print(f"Loaded checkpoint '{filepath}' (epoch {epoch}, seed={saved_seed})")
    return epoch, model, optimizer


def compute_wer(hypotheses, references):
    hyp_strs = [" ".join(h) if len(h) > 0 else "<empty>" for h in hypotheses]
    ref_strs = [" ".join(r) if len(r) > 0 else "<empty>" for r in references]
    return jiwer.wer(ref_strs, hyp_strs)


def train_step(
    model,
    optimizer,
    frames,
    frame_lengths,
    targets,
    target_lengths,
    criterion,
    kl_weight=KL_WEIGHT,
    grad_clip=GRAD_CLIP,
    device=DEVICE,
):
    """
    Single training step on a batch.
    """
    optimizer.zero_grad()

    frames_permuted = frames.permute(0, 3, 1, 2)  # [B, C, T, V]

    beta_dist = torch.distributions.beta.Beta(2.0, 2.0)
    # dynamic_keep_prob = beta_dist.sample().item()
    # dynamic_keep_prob = max(0.1, min(0.9, dynamic_keep_prob))
    dynamic_keep_prob = 0.8


    outputs = model(frames_permuted, frame_lengths, keep_prob=dynamic_keep_prob)

    loss_dict = compute_cosign_loss(
        outputs,
        targets,
        target_lengths,
        criterion,
        kl_weight=kl_weight,
        keep_prob=dynamic_keep_prob,
    )

    loss = loss_dict["total"]
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    return loss.item(), dynamic_keep_prob


def compute_cosign_loss(
    outputs, targets, target_lengths, criterion, kl_weight=KL_WEIGHT, keep_prob=0.8
):
    """Calculates CTC Loss and Bidirectional KL Divergence across branches."""
    total_loss = 0.0
    ctc_losses = []

    branches_to_train = ["phi"] if keep_prob == 1.0 else ["phi", "phi_inv"]

    for branch in branches_to_train:
        branch_out = outputs[branch]
        for head in ["aux_logits", "main_logits"]:
            logits = branch_out[head]
            logit_lengths = branch_out["logit_lengths"]
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)

            loss = criterion(log_probs, targets, logit_lengths, target_lengths)
            ctc_losses.append(loss)

    ctc_loss = torch.stack(ctc_losses).mean()
    total_loss = ctc_loss
    kl_loss = torch.tensor(0.0, device=ctc_loss.device)

    if keep_prob < 1.0:
        T_prime = outputs["phi"]["aux_logits"].size(1)

        aux_phi = F.log_softmax(outputs["phi"]["aux_logits"], dim=-1)
        aux_phibar = F.log_softmax(outputs["phi_inv"]["aux_logits"], dim=-1)
        aux_phi_soft = F.softmax(aux_phi, dim=-1)
        aux_phibar_soft = F.softmax(aux_phibar, dim=-1)

        kl_aux = F.kl_div(aux_phi, aux_phibar_soft, reduction="batchmean") + F.kl_div(
            aux_phibar, aux_phi_soft, reduction="batchmean"
        )
        kl_aux = kl_aux / T_prime

        main_phi = F.log_softmax(outputs["phi"]["main_logits"], dim=-1)
        main_phibar = F.log_softmax(outputs["phi_inv"]["main_logits"], dim=-1)
        main_phi_soft = F.softmax(main_phi, dim=-1)
        main_phibar_soft = F.softmax(main_phibar, dim=-1)

        kl_main = F.kl_div(main_phi, main_phibar_soft, reduction="batchmean") + F.kl_div(
            main_phibar, main_phi_soft, reduction="batchmean"
        )
        kl_main = kl_main / T_prime

        kl_loss = (kl_aux + kl_main) * 0.5 * kl_weight
        total_loss += kl_loss

    return {
        "total": total_loss,
        "ctc": ctc_loss,
        "kl": kl_loss,
    }


def evaluate(model, dataloader, criterion, id2gloss, device):
    """Greedy decode evaluation."""
    model.eval()
    total_loss = 0.0
    all_hyps_phi, all_refs = [], []

    with torch.no_grad():
        for batch in dataloader:
            frames = batch["frames"].to(device).permute(0, 3, 1, 2)
            frame_lengths = batch["frame_lengths"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            outputs = model(frames, frame_lengths, keep_prob=1.0)
            loss_dict = compute_cosign_loss(
                outputs, targets, target_lengths, criterion, kl_weight=0.0, keep_prob=1.0
            )
            total_loss += loss_dict["total"].item()

            logits = outputs["phi"]["main_logits"]
            seq_lengths = outputs["phi"]["logit_lengths"]
            preds = torch.argmax(logits, dim=-1)

            for i in range(targets.size(0)):
                hyp = []
                prev_token = -1
                for t in range(seq_lengths[i]):
                    token = preds[i, t].item()
                    if token != 0 and token != prev_token:
                        hyp.append(token)
                    prev_token = token
                all_hyps_phi.append([id2gloss.get(v, "") for v in hyp])

                ref = [
                    id2gloss.get(v.item(), "")
                    for v in targets[i][: target_lengths[i]]
                    if v.item() != 0
                ]
                all_refs.append(ref)

    avg_wer = compute_wer(all_hyps_phi, all_refs)
    for i in range(min(2, len(all_refs))):
        print(f"Target: {' '.join(all_refs[i])}")
        print(f"Pred:   {' '.join(all_hyps_phi[i])}\n")

    return total_loss / len(dataloader), avg_wer


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    set_seed(SEED, deterministic=DETERMINISTIC)
    print(f"Using seed: {SEED} | Deterministic: {DETERMINISTIC}")

    if REPRODUCING_RUN:
        run_id = f"{original_run_id}_reproduction"
    else:
        run_id = f"cosign_{time.strftime('%Y%m%d_%H%M%S')}"

    tb_writer = SummaryWriter(os.path.join(LOG_DIR, run_id)) if LOG_TENSORBOARD else None
    if LOG_WANDB:
        wandb.init(project="cosign-sign-language", name=run_id, config=config, dir=LOG_DIR)

    print("Building vocabulary")
    dataset_type = config.get("data", {}).get("dataset", "phoenix")

    if dataset_type == "pjm":
        annotation_dir = config["data"]["annotation_dir"]
        gloss2id, id2gloss = build_pjm_vocab(annotation_dir, [ANN_TRAIN, ANN_DEV])
        num_classes = len(gloss2id)

        print("Loading PJM datasets")
        train_dataset = PJMDataset(DATA_DIR_TRAIN, annotation_dir, ANN_TRAIN, gloss2id)
        dev_dataset = PJMDataset(DATA_DIR_DEV, annotation_dir, ANN_DEV, gloss2id, split="test")
        collate_fn = pjm_ctc_collate_fn
    else:
        gloss2id, id2gloss = build_phoenix_vocab([ANN_TRAIN, ANN_DEV])
        num_classes = len(gloss2id)

        print("Loading Phoenix datasets")
        train_dataset = PhoenixDataset(DATA_DIR_TRAIN, ANN_TRAIN, gloss2id)
        dev_dataset = PhoenixDataset(DATA_DIR_DEV, ANN_DEV, gloss2id)
        collate_fn = phoenix_ctc_collate_fn

    g = torch.Generator()
    g.manual_seed(SEED)
    num_classes = len(gloss2id)

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

    total_train_batches = len(train_loader)
    print(f"Initializing model on {DEVICE}")
    model = CoSign1SModel(num_classes=num_classes, dropout=MODEL_DROPOUT)
    model.to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    run_manifest_path = os.path.join(CHECKPOINT_DIR, f"{run_id}_training_run.json")
    extra_info = {
        "run_id": run_id,
        "num_classes": num_classes,
        "train_samples": len(train_dataset),
        "dev_samples": len(dev_dataset),
        "device": str(DEVICE),
    }
    if REPRODUCING_RUN:
        extra_info.update(
            {
                "is_reproduction": True,
                "original_run_id": original_run_id,
                "original_manifest_path": args.reproduce_run,
            }
        )
    save_training_run(
        run_manifest_path,
        config,
        SEED,
        DETERMINISTIC,
        extra_info=extra_info,
    )

    start_epoch = 0
    latest_ckpt = os.path.join(CHECKPOINT_DIR, "latest.pth")
    if os.path.exists(latest_ckpt):
        start_epoch, _, _ = load_checkpoint(latest_ckpt, model, optimizer)

    best_wer = float("inf")

    for epoch in range(start_epoch, EPOCHS):
        current_lr = LEARNING_RATE
        if epoch >= OPTIMIZER_MILESTONES[0]:
            current_lr *= OPTIMIZER_GAMMA
        if len(OPTIMIZER_MILESTONES) > 1 and epoch >= OPTIMIZER_MILESTONES[1]:
            current_lr *= OPTIMIZER_GAMMA

        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        model.train()
        total_train_loss = 0
        num_batches = 0

        print(f"Epoch {epoch + 1:3d} | Learning rate: {current_lr:.2e}")

        for batch_idx, batch in enumerate(train_loader):
            frames = batch["frames"].to(DEVICE)  # [B, T, V, C]
            frame_lengths = batch["frame_lengths"].to(DEVICE)
            targets = batch["targets"].to(DEVICE)
            target_lengths = batch["target_lengths"].to(DEVICE)

            loss_orig, keep_prob_orig = train_step(
                model,
                optimizer,
                frames,
                frame_lengths,
                targets,
                target_lengths,
                criterion,
                KL_WEIGHT,
                GRAD_CLIP,
                DEVICE,
            )
            total_train_loss += loss_orig
            num_batches += 1

            if MIRROR_DATA:
                frames_mirrored = mirror_batch(frames, frame_lengths)
                loss_mirrored, keep_prob_mirrored = train_step(
                    model,
                    optimizer,
                    frames_mirrored,
                    frame_lengths,
                    targets,
                    target_lengths,
                    criterion,
                    KL_WEIGHT,
                    GRAD_CLIP,
                    DEVICE,
                )
                total_train_loss += loss_mirrored
                num_batches += 1

            if batch_idx % 100 == 0:
                if MIRROR_DATA:
                    print(
                        f"  Batch {batch_idx + 1}/{len(train_loader)} | Orig: {loss_orig:.4f} | Mirr: {loss_mirrored:.4f} | keep_prob: {keep_prob_orig:.2f}/{keep_prob_mirrored:.2f}"
                    )
                else:
                    print(
                        f"  Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss_orig:.4f} | keep_prob: {keep_prob_orig:.2f}"
                    )

        avg_train_loss = total_train_loss / num_batches
        val_loss, avg_wer = evaluate(model, dev_loader, criterion, id2gloss, DEVICE)

        epoch_metrics = {
            "val/loss": val_loss,
            "val/wer": avg_wer,
            "train/avg_loss": avg_train_loss,
        }
        if tb_writer:
            for k, v in epoch_metrics.items():
                tb_writer.add_scalar(k, v, epoch + 1)
        if LOG_WANDB:
            wandb.log(epoch_metrics, step=epoch + 1)

        print(
            f"Epoch {epoch + 1} done. Train: {avg_train_loss:.4f}, Val: {val_loss:.4f}, WER: {avg_wer:.3f}"
        )

        save_checkpoint(model, optimizer, epoch + 1, gloss2id, val_loss, SEED, latest_ckpt)

        if avg_wer < best_wer:
            best_wer = avg_wer
            best_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            save_checkpoint(model, optimizer, epoch + 1, gloss2id, val_loss, SEED, best_path)
            print(f"-> NEW BEST WER: {best_wer:.3f}")
            extra_info = {
                "run_id": run_id,
                "num_classes": num_classes,
                "train_samples": len(train_dataset),
                "dev_samples": len(dev_dataset),
                "device": str(DEVICE),
                "best_wer": best_wer,
                "best_wer_epoch": epoch + 1,
                "best_train_loss": avg_train_loss,
                "best_val_loss": val_loss,
            }
            if REPRODUCING_RUN:
                extra_info.update(
                    {
                        "is_reproduction": True,
                        "original_run_id": original_run_id,
                        "original_manifest_path": args.reproduce_run,
                    }
                )
            save_training_run(
                run_manifest_path,
                config,
                SEED,
                DETERMINISTIC,
                extra_info=extra_info,
            )

    if tb_writer:
        tb_writer.close()
    if LOG_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
