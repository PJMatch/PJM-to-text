from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from phoenix_dataloader import (
    PhoenixDataset,
    build_gloss_vocab,
    phoenix_ctc_collate_fn,
)
from model import CLSROfflineModel
from edges import pose_edges, hand_edges

import subprocess
import time


def greedy_ctc_decode(log_probs, output_lengths, blank_id=0):
    pred_ids = log_probs.argmax(dim=-1)  # (B, T)
    decoded = []

    for seq, seq_len in zip(pred_ids, output_lengths):
        seq = seq[:seq_len].tolist()

        collapsed = []
        prev = None
        for tok in seq:
            if tok != prev:
                collapsed.append(tok)
            prev = tok

        collapsed = [t for t in collapsed if t != blank_id]
        decoded.append(collapsed)

    return decoded


def save_checkpoint(path, model, optimizer, scaler, epoch, best_dev_loss, gloss2id, id2gloss):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "best_dev_loss": best_dev_loss,
            "gloss2id": gloss2id,
            "id2gloss": id2gloss,
        },
        path,
    )

def compute_losses(
    outputs,
    targets,
    target_lengths,
    ctc_loss,
    aux_weight=0.3,
    cm_weight=0.3,
    cr_weight=0.1,
):
    def to_ctc_shape(x):
        return x.permute(1, 0, 2).contiguous()  # (T, B, C)

    loss_main = ctc_loss(
        to_ctc_shape(outputs["main_log_probs"]),
        targets,
        outputs["main_lengths"],
        target_lengths,
    )

    loss_aux = ctc_loss(
        to_ctc_shape(outputs["aux_log_probs"]),
        targets,
        outputs["aux_lengths"],
        target_lengths,
    )

    loss_cm1 = ctc_loss(
        to_ctc_shape(outputs["cm1_log_probs"]),
        targets,
        outputs["cm1_lengths"],
        target_lengths,
    )

    loss_cm2 = ctc_loss(
        to_ctc_shape(outputs["cm2_log_probs"]),
        targets,
        outputs["cm2_lengths"],
        target_lengths,
    )

    loss_cr = outputs["cr_loss"]

    total = (
        loss_main
        + aux_weight * loss_aux
        + cm_weight * (loss_cm1 + loss_cm2)
        + cr_weight * loss_cr
    )

    stats = {
        "total": total.detach(),
        "main": loss_main.detach(),
        "aux": loss_aux.detach(),
        "cm1": loss_cm1.detach(),
        "cm2": loss_cm2.detach(),
        "cr": loss_cr.detach(),
    }
    return total, stats



def train_one_epoch(
    model,
    loader,
    ctc_loss,
    optimizer,
    scaler,
    device,
    use_amp=True,
    accum_steps=1,
    grad_clip=5.0,
    aux_ctc_weight=0.3,
    cr_weight=0.1,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    running = {k: 0.0 for k in ["total", "main", "aux1", "aux2", "cr"]}
    num_batches = len(loader)
    amp_enabled = use_amp and device.startswith("cuda")
    effective_steps = 0

    for step, batch in enumerate(loader, start=1):

        frames = batch["frames"].to(device, non_blocking=True)
        frame_lengths = batch["frame_lengths"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        target_lengths = batch["target_lengths"].to(device, non_blocking=True)

        try:
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                outputs = model(frames, frame_lengths)
                total_loss, loss_dict = compute_losses(
                    outputs=outputs,
                    targets=targets,
                    target_lengths=target_lengths,
                    ctc_loss=ctc_loss,
                    aux_ctc_weight=aux_ctc_weight,
                    cr_weight=cr_weight,
                )
                loss_for_backward = total_loss / accum_steps

            if amp_enabled:
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("[OOM] Skipping batch, clearing CUDA cache.")
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                optimizer.zero_grad(set_to_none=True)
                continue
            else:
                raise

        if step % accum_steps == 0 or step == num_batches:
            if amp_enabled:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            effective_steps += 1

        for k in running:
            running[k] += float(loss_dict[k].item())

    denom = max(effective_steps, 1)
    for k in running:
        running[k] /= denom

    return running


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    ctc_loss,
    device,
    use_amp=True,
    aux_ctc_weight=0.3,
    cr_weight=0.1,
    blank_id=0,
):
    model.eval()

    running = {
        "total": 0.0,
        "main": 0.0,
        "aux1": 0.0,
        "aux2": 0.0,
        "cr": 0.0,
    }

    all_preds = []
    all_targets = []

    amp_enabled = use_amp and device.startswith("cuda")

    for batch in loader:
        frames = batch["frames"].to(device, non_blocking=True)
        frame_lengths = batch["frame_lengths"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        target_lengths = batch["target_lengths"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            outputs = model(frames, frame_lengths)
            total_loss, loss_dict = compute_losses(
                outputs=outputs,
                targets=targets,
                target_lengths=target_lengths,
                ctc_loss=ctc_loss,
                aux_ctc_weight=aux_ctc_weight,
                cr_weight=cr_weight,
            )

        for k in running:
            running[k] += float(loss_dict[k].item())

        preds = greedy_ctc_decode(
            outputs["main_log_probs"],
            outputs["output_lengths"],
            blank_id=blank_id,
        )
        all_preds.extend(preds)

        for tgt, tgt_len in zip(targets, target_lengths):
            all_targets.append(tgt[:tgt_len].tolist())

    for k in running:
        running[k] /= max(len(loader), 1)

    return running, all_preds, all_targets


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = True
    accum_steps = 2
    batch_size = 4
    num_epochs = 40
    lr = 4e-4
    weight_decay = 1e-4
    aux_ctc_weight = 0.3
    cr_weight = 0.1
    blank_id = 0

    train_ann = "./annotations/PHOENIX-2014-T.train.corpus.csv"
    dev_ann = "./annotations/PHOENIX-2014-T.dev.corpus.csv"
    train_dir = "./pheonix-dataset/train/"
    dev_dir = "./pheonix-dataset/dev/"

    gloss2id, id2gloss = build_gloss_vocab([train_ann, dev_ann])

    train_dataset = PhoenixDataset(train_dir, train_ann, gloss2id)
    dev_dataset = PhoenixDataset(dev_dir, dev_ann, gloss2id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=phoenix_ctc_collate_fn,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=phoenix_ctc_collate_fn,
    )

    model = CLSROfflineModel(
        num_classes=len(gloss2id) - 1,
        pose_edges=pose_edges,
        hand_edges=hand_edges,
        dropout=0.2,
        bidirectional=False,
    ).to(device)

    ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=True)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[20, 35],
        gamma=0.1,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(
        use_amp and device.startswith("cuda")))

    best_dev_loss = float("inf")

    print(f"Device: {device}")
    if device.startswith("cuda"):
        print(torch.cuda.get_device_name(0))
        total_mem_gb = torch.cuda.get_device_properties(
            0).total_memory / (1024 ** 3)
        print(f"GPU memory: {total_mem_gb:.1f} GB")

    for epoch in range(1, num_epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            ctc_loss=ctc_loss,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            accum_steps=accum_steps,
            grad_clip=5.0,
            aux_ctc_weight=aux_ctc_weight,
            cr_weight=cr_weight,
        )

        dev_stats, dev_preds, dev_targets = validate_one_epoch(
            model=model,
            loader=dev_loader,
            ctc_loss=ctc_loss,
            device=device,
            use_amp=use_amp,
            aux_ctc_weight=aux_ctc_weight,
            cr_weight=cr_weight,
            blank_id=blank_id,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d} | "
            f"lr={current_lr:.2e} | "
            f"train_total={train_stats['total']:.4f} | "
            f"train_main={train_stats['main']:.4f} | "
            f"train_aux1={train_stats['aux1']:.4f} | "
            f"train_aux2={train_stats['aux2']:.4f} | "
            f"train_cr={train_stats['cr']:.4f} | "
            f"dev_total={dev_stats['total']:.4f} | "
            f"dev_main={dev_stats['main']:.4f}"
        )

        save_checkpoint(
            path="checkpoints/latest.pt",
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            best_dev_loss=best_dev_loss,
            gloss2id=gloss2id,
            id2gloss=id2gloss,
        )

        if dev_stats["total"] < best_dev_loss:
            best_dev_loss = dev_stats["total"]
            save_checkpoint(
                path="checkpoints/best.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                best_dev_loss=best_dev_loss,
                gloss2id=gloss2id,
                id2gloss=id2gloss,
            )
            print(f"Saved new best checkpoint at epoch {epoch}")

        scheduler.step()

    print("Training finished.")


if __name__ == "__main__":
    main()
