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
from pjm_dataloader_cslr import PJMDataset, build_gloss_vocab as build_pjm_vocab, pjm_ctc_collate_fn


def parse_args():
    """Parses command-line arguments, including options to reproduce a specific CSLR run."""
    parser = argparse.ArgumentParser(description="Train CoSign sign language recognition model")
    parser.add_argument(
        "--reproduce_run",
        type=str,
        default=None,
        help="Path to training_run.json file to reproduce a previous run",
    )
    return parser.parse_args()


def mirror_batch(frames, frame_lengths):
    """Applies horizontal mirroring to the skeleton frames for data augmentation."""
    mirrored = frames.clone()
    mirrored[:, :, :, 0] *= -1
    return mirrored


def load_config(config_path="config.yaml"):
    """Loads the YAML configuration settings."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_training_run_manifest(manifest_path):
    """Loads a JSON manifest file to recreate the environment of a previous training run."""
    with open(manifest_path, "r") as f:
        run_info = json.load(f)
    return run_info


def set_seed(seed, deterministic=True):
    """Forces determinism across random, numpy, and torch environments."""
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
    """Ensures each dataloader worker has a unique, deterministic seed."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def save_training_run(filepath, config, seed, deterministic, extra_info=None):
    """Saves run details (hyperparameters, seeds) to JSON for future reproducibility."""
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
KL_WEIGHT = float(config["training"].get("kl_weight", 0.0))
MIRROR_DATA = config["training"].get("mirror_data", False)
KEEP_PROB = config["training"].get("keep_prob", 0.8)
INIT_WEIGHTS = config["training"].get("init_weights", None)
KL_WARMUP_START = config["training"].get("kl_warmup_start", 5)
KL_WARMUP_END = config["training"].get("kl_warmup_end", 10)

if config["system"]["device"] == "auto":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device(config["system"]["device"])

CHECKPOINT_DIR = config["system"]["checkpoint_dir"]
DATA_DIR = config["data"]["data_dir"]
ANN_TRAIN = config["data"]["train_ann"]
ANN_DEV = config["data"]["dev_ann"]
NUM_WORKERS = config["data"]["num_workers"]
PIN_MEMORY = config["data"]["pin_memory"]

MODEL_DROPOUT = config["model"]["dropout"]

LOG_TENSORBOARD = config["logging"]["tensorboard"]
LOG_WANDB = config["logging"]["wandb"] and WANDB_AVAILABLE
LOG_DIR = config["logging"]["log_dir"]
LOG_INTERVAL = config["logging"].get("log_interval", 50)
HISTOGRAM_INTERVAL = config["logging"].get("histogram_interval", 5)

OPTIMIZER_MILESTONES = config["optimizer"]["milestones"]
OPTIMIZER_GAMMA = float(config["optimizer"]["gamma"])


def save_checkpoint(model, optimizer, epoch, gloss2id, val_loss, seed, filepath):
    """Saves model weights, optimizer, and validation metrics for the CSLR model."""
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
    """Loads the checkpoint to resume training from a specific epoch."""
    checkpoint = torch.load(filepath, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    val_loss = checkpoint.get("val_loss", float("inf"))
    saved_seed = checkpoint.get("seed", "unknown")
    print(f"Loaded checkpoint '{filepath}' (epoch {epoch}, seed={saved_seed})")
    return epoch, model, optimizer


def compute_wer(hypotheses, references):
    """Calculates the WER using the jiwer library."""
    hyp_strs = [" ".join(h) if len(h) > 0 else "<empty>" for h in hypotheses]
    ref_strs = [" ".join(r) if len(r) > 0 else "<empty>" for r in references]
    return jiwer.wer(ref_strs, hyp_strs)


def _masked_kl_symmetric(p_logits, q_logits, lengths):
    """Computes Symmetric KL Divergence between two branches, masked to ignore padded timesteps."""
    p_log = F.log_softmax(p_logits, dim=-1)
    q_log = F.log_softmax(q_logits, dim=-1)
    p_soft = p_log.exp()
    q_soft = q_log.exp()

    kl_pq = (p_soft * (p_log - q_log)).sum(dim=-1)
    kl_qp = (q_soft * (q_log - p_log)).sum(dim=-1)
    kl_per_token = kl_pq + kl_qp

    B, T = kl_per_token.shape
    time_idx = torch.arange(T, device=kl_per_token.device).unsqueeze(0)
    mask = (time_idx < lengths.unsqueeze(1)).float()

    valid_count = mask.sum().clamp_min(1.0)
    return (kl_per_token * mask).sum() / valid_count


def get_keep_prob(epoch):
    """Keep probability (can be made dynamic in the future)."""
    return KEEP_PROB


def get_kl_weight(epoch):
    """KL active from ep KL_WARMUP_START, linear to ep KL_WARMUP_END."""
    if epoch < KL_WARMUP_START:
        return 0.0
    if epoch < KL_WARMUP_END:
        return KL_WEIGHT * ((epoch - KL_WARMUP_START) / (KL_WARMUP_END - KL_WARMUP_START))
    return KL_WEIGHT


def train_step(
    model,
    optimizer,
    frames,
    frame_lengths,
    targets,
    target_lengths,
    criterion,
    kl_weight=KL_WEIGHT,
    keep_prob=KEEP_PROB,
    grad_clip=GRAD_CLIP,
    device=DEVICE,
):
    """Executes a single forward and backward pass for a training batch."""
    optimizer.zero_grad()

    frames_permuted = frames.permute(0, 3, 1, 2)  # [B, C, T, V]

    beta_dist = torch.distributions.beta.Beta(2.0, 2.0)
    # dynamic_keep_prob = beta_dist.sample().item()
    # dynamic_keep_prob = max(0.1, min(0.9, dynamic_keep_prob))
    dynamic_keep_prob = keep_prob

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

    return loss_dict, dynamic_keep_prob


def _log_batch_tb(tb_writer, loss_dict, keep_prob, global_step):
    """Logs individual batch metrics (CTC loss, KL divergence) to TensorBoard."""
    if tb_writer is None:
        return
    tb_writer.add_scalar("batch/total_loss", loss_dict["total"].item(), global_step)
    tb_writer.add_scalar("batch/ctc_loss", loss_dict["ctc"].item(), global_step)
    tb_writer.add_scalar("batch/kl_loss", loss_dict["kl"].item(), global_step)
    tb_writer.add_scalar("batch/ctc_aux", loss_dict["ctc_aux"].item(), global_step)
    tb_writer.add_scalar("batch/ctc_main", loss_dict["ctc_main"].item(), global_step)
    tb_writer.add_scalar("batch/keep_prob", keep_prob, global_step)


def _log_epoch_histograms(tb_writer, model, epoch):
    """Logs network weight and gradient distributions to TensorBoard."""
    if tb_writer is None:
        return
    for name, param in model.named_parameters():
        if param.ndim >= 2:  # skip biases, scale, etc.
            tb_writer.add_histogram(f"weights/{name}", param.data, epoch)
            if param.grad is not None:
                tb_writer.add_histogram(f"grads/{name}", param.grad, epoch)
    # Log gloss_head scale separately (important for training dynamics)
    if hasattr(model, "gloss_head") and hasattr(model.gloss_head, "scale"):
        tb_writer.add_scalar("model/gloss_head_scale", model.gloss_head.scale.item(), epoch)


def compute_cosign_loss(
    outputs, targets, target_lengths, criterion, kl_weight=KL_WEIGHT, keep_prob=0.8
):
    """
    Calculates the combined loss for the CoSign architecture.
    Includes Connectionist Temporal Classification (CTC) Loss and Bidirectional KL Divergence.
    """
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

    if keep_prob == 1.0:
        ctc_loss = torch.stack(ctc_losses).sum()
    else:
        ctc_loss = 0.5 * torch.stack(ctc_losses).sum()
    total_loss = ctc_loss
    kl_loss = torch.tensor(0.0, device=ctc_loss.device)

    if keep_prob < 1.0:
        logit_lengths = outputs["phi"]["logit_lengths"]
        kl_aux = _masked_kl_symmetric(
            outputs["phi"]["aux_logits"],
            outputs["phi_inv"]["aux_logits"],
            logit_lengths,
        )
        kl_main = _masked_kl_symmetric(
            outputs["phi"]["main_logits"],
            outputs["phi_inv"]["main_logits"],
            logit_lengths,
        )
        kl_loss = (kl_aux + kl_main) * kl_weight
        total_loss += kl_loss

    return {
        "total": total_loss,
        "ctc": ctc_loss,
        "ctc_aux": ctc_losses[0].detach(),
        "ctc_main": ctc_losses[1].detach() if len(ctc_losses) > 1 else torch.tensor(0.0, device=ctc_loss.device),
        "kl": kl_loss,
    }


def evaluate(model, dataloader, criterion, id2gloss, device):
    """Evaluates the CSLR model using Greedy CTC Decoding and WER"""
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

            if not hasattr(evaluate, "_diag_done"):
                evaluate._diag_done = True
                seq_len_0 = seq_lengths[0].item()
                max_sim = logits[0, :seq_len_0].max(dim=-1).values
                blank_frac = (preds[0, :seq_len_0] == 0).float().mean().item()
                non_blank = (preds[0, :seq_len_0] != 0).sum().item()
                scale_val = model.gloss_head.scale.item() if hasattr(model, "gloss_head") else float("nan")
                print(f"[DIAG] frames: min={frames.min().item():.3f} max={frames.max().item():.3f} "
                      f"mean={frames.mean().item():.3f} std={frames.std().item():.3f}")
                print(f"[DIAG] max_sim: mean={max_sim.mean().item():.3f} "
                      f"max={max_sim.max().item():.3f} scale={scale_val:.2f}")
                print(f"[DIAG] blank_frac={blank_frac:.3f} non_blank_tokens={non_blank}/{seq_len_0}")
                print(f"[DIAG] ctc_aux={loss_dict['ctc_aux'].item():.3f} ctc_main={loss_dict['ctc_main'].item():.3f}")

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

    examples = []
    for i in range(min(5, len(all_refs))):
        examples.append((all_refs[i], all_hyps_phi[i]))

    return total_loss / len(dataloader), avg_wer, examples

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
    annotation_dir = config["data"]["annotation_dir"]
    gloss2id, id2gloss = build_pjm_vocab(annotation_dir, [ANN_TRAIN, ANN_DEV])
    num_classes = len(gloss2id)

    print("Loading PJM datasets")
    train_dataset = PJMDataset(DATA_DIR, annotation_dir, ANN_TRAIN, gloss2id)
    dev_dataset = PJMDataset(DATA_DIR, annotation_dir, ANN_DEV, gloss2id, split="test")
    collate_fn = pjm_ctc_collate_fn

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

    print(f"Initializing model on {DEVICE}")
    model = CoSign1SModel(num_classes=num_classes, dropout=MODEL_DROPOUT)
    model.to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    if INIT_WEIGHTS and os.path.exists(INIT_WEIGHTS):
        print(f"Loading initial weights from: {INIT_WEIGHTS}")
        ckpt = torch.load(INIT_WEIGHTS, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  -> Loaded weights from epoch {ckpt.get('epoch', '?')}, "
              f"val_loss={ckpt.get('val_loss', '?')}")
        print(f"  -> Optimizer is FRESH (LR={LEARNING_RATE})")
        print(f"  -> Epoch counter starts at 0 (milestones={OPTIMIZER_MILESTONES})")

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
    global_step = 0

    for epoch in range(start_epoch, EPOCHS):
        current_lr = LEARNING_RATE
        if epoch >= OPTIMIZER_MILESTONES[0]:
            current_lr *= OPTIMIZER_GAMMA
        if len(OPTIMIZER_MILESTONES) > 1 and epoch >= OPTIMIZER_MILESTONES[1]:
            current_lr *= OPTIMIZER_GAMMA

        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        current_kl_weight = get_kl_weight(epoch)
        current_keep_prob = get_keep_prob(epoch)

        model.train()
        total_train_loss = 0
        num_batches = 0

        print(f"Epoch {epoch + 1:3d} | LR: {current_lr:.2e} | "
              f"keep_prob: {current_keep_prob:.3f} | kl_weight: {current_kl_weight:.3f}")

        for batch_idx, batch in enumerate(train_loader):
            frames = batch["frames"].to(DEVICE)  # [B, T, V, C]
            frame_lengths = batch["frame_lengths"].to(DEVICE)
            targets = batch["targets"].to(DEVICE)
            target_lengths = batch["target_lengths"].to(DEVICE)

            loss_dict_orig, keep_prob_orig = train_step(
                model,
                optimizer,
                frames,
                frame_lengths,
                targets,
                target_lengths,
                criterion,
                current_kl_weight,
                current_keep_prob,
                GRAD_CLIP,
                DEVICE,
            )
            total_train_loss += loss_dict_orig["total"].item()
            num_batches += 1
            global_step += 1

            if MIRROR_DATA:
                frames_mirrored = mirror_batch(frames, frame_lengths)
                loss_dict_mirrored, keep_prob_mirrored = train_step(
                    model,
                    optimizer,
                    frames_mirrored,
                    frame_lengths,
                    targets,
                    target_lengths,
                    criterion,
                    current_kl_weight,
                    current_keep_prob,
                    GRAD_CLIP,
                    DEVICE,
                )
                total_train_loss += loss_dict_mirrored["total"].item()
                num_batches += 1
                global_step += 1

            # Per-batch TB logging
            if tb_writer and batch_idx % LOG_INTERVAL == 0:
                _log_batch_tb(tb_writer, loss_dict_orig, keep_prob_orig, global_step)

            if batch_idx % 100 == 0:
                if MIRROR_DATA:
                    print(
                        f"  Batch {batch_idx + 1}/{len(train_loader)} | Orig: {loss_dict_orig['total'].item():.4f} | Mirr: {loss_dict_mirrored['total'].item():.4f} | keep_prob: {keep_prob_orig:.2f}/{keep_prob_mirrored:.2f}"
                    )
                else:
                    print(
                        f"  Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss_dict_orig['total'].item():.4f} | keep_prob: {keep_prob_orig:.2f}"
                    )

        avg_train_loss = total_train_loss / num_batches
        val_loss, avg_wer, eval_examples = evaluate(model, dev_loader, criterion, id2gloss, DEVICE)

        epoch_metrics = {
            "val/loss": val_loss,
            "val/wer": avg_wer,
            "train/avg_loss": avg_train_loss,
            "train/lr": current_lr,
            "train/kl_weight": current_kl_weight,
        }
        if tb_writer:
            for k, v in epoch_metrics.items():
                tb_writer.add_scalar(k, v, epoch + 1)

            if (epoch + 1) % HISTOGRAM_INTERVAL == 0 or epoch == 0:
                _log_epoch_histograms(tb_writer, model, epoch + 1)

        if LOG_WANDB:
            wandb.log(epoch_metrics, step=epoch + 1)

        if tb_writer:
            for i, (ref, hyp) in enumerate(eval_examples):
                ref_str = " ".join(ref) if ref else "<empty>"
                hyp_str = " ".join(hyp) if hyp else "<empty>"
                wer_i = jiwer.wer(ref_str, hyp_str)
                tb_writer.add_text(
                    f"val/example_{i}",
                    f"**Target:** {ref_str}\n\n**Pred:** {hyp_str}\n\n**WER:** {wer_i:.2f}",
                    epoch + 1,
                )

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
