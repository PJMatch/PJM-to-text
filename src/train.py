import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio.models.decoder as decoder
import tempfile
import jiwer
import torch.nn.functional as F
import yaml

from model import CoSign1SModel
from phoenix_dataloader import PhoenixDataset, phoenix_ctc_collate_fn, build_gloss_vocab


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


config = load_config()

EPOCHS = config["training"]["epochs"]
BATCH_SIZE = config["training"]["batch_size"]
LEARNING_RATE = float(config["training"]["learning_rate"])
WEIGHT_DECAY = float(config["training"]["weight_decay"])
GRAD_CLIP = float(config["training"]["grad_clip"])
KEEP_PROB = float(config["training"]["keep_prob"])
KL_WEIGHT = float(config["training"]["kl_weight"])

if config["system"]["device"] == "auto":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device(config["system"]["device"])

CHECKPOINT_DIR = config["system"]["checkpoint_dir"]

DATA_DIR_TRAIN = config["data"]["train_dir"]
DATA_DIR_DEV = config["data"]["dev_dir"]
ANN_TRAIN = config["data"]["train_ann"]
ANN_DEV = config["data"]["dev_ann"]
NUM_WORKERS = config["data"]["num_workers"]
PIN_MEMORY = config["data"]["pin_memory"]

MODEL_DROPOUT = config["model"]["dropout"]

OPTIMIZER_MILESTONES = config["optimizer"]["milestones"]
OPTIMIZER_GAMMA = float(config["optimizer"]["gamma"])


def save_checkpoint(model, optimizer, epoch, gloss2id, val_loss, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'gloss2id': gloss2id,
        'val_loss': val_loss
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer):
    checkpoint = torch.load(filepath, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_loss = checkpoint.get('val_loss', float('inf'))
    print(f"Loaded checkpoint '{filepath}' (Resuming from epoch {epoch})")
    return epoch, model, optimizer


def compute_wer(hypotheses, references):
    hyp_strs = [" ".join(h) if len(h) > 0 else "<empty>" for h in hypotheses]
    ref_strs = [" ".join(r) if len(r) > 0 else "<empty>" for r in references]
    return jiwer.wer(ref_strs, hyp_strs)


def compute_cosign_loss(outputs, targets, target_lengths, criterion, kl_weight=KL_WEIGHT, keep_prob=0.8):
    """Calculates CTC and KL-Divergence dynamically based on keep_prob."""
    total_loss = 0.0
    ctc_losses = []

    # If keep_prob is 1.0 (warmup), inverse branch is garbage. Don't penalize it.
    branches_to_train = ["phi"] if keep_prob == 1.0 else ["phi", "phi_inv"]

    for branch in branches_to_train:
        branch_out = outputs[branch]
        for head in ["aux_logits", "main_logits"]:
            logits = branch_out[head]
            logit_lengths = branch_out["logit_lengths"]
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
            ctc_losses.append(criterion(log_probs, targets,
                              logit_lengths, target_lengths))

    total_loss += torch.stack(ctc_losses).mean()

    if keep_prob < 1.0:
        aux_phi = F.log_softmax(outputs["phi"]["aux_logits"], dim=-1)
        aux_phibar = F.log_softmax(outputs["phi_inv"]["aux_logits"], dim=-1)
        aux_phi_soft = F.softmax(aux_phi, dim=-1)
        aux_phibar_soft = F.softmax(aux_phibar, dim=-1)

        kl_aux = F.kl_div(aux_phi, aux_phibar_soft, reduction='batchmean') + \
            F.kl_div(aux_phibar, aux_phi_soft, reduction='batchmean')

        main_phi = F.log_softmax(outputs["phi"]["main_logits"], dim=-1)
        main_phibar = F.log_softmax(outputs["phi_inv"]["main_logits"], dim=-1)
        main_phi_soft = F.softmax(main_phi, dim=-1)
        main_phibar_soft = F.softmax(main_phibar, dim=-1)

        kl_main = F.kl_div(main_phi, main_phibar_soft, reduction='batchmean') + \
            F.kl_div(main_phibar, main_phi_soft, reduction='batchmean')

        kl_loss = (kl_aux + kl_main) * 0.5 * kl_weight
        total_loss += kl_loss

    return total_loss


def evaluate(model, dataloader, criterion, ctc_decoder_obj, id2gloss, device):
    model.eval()
    total_loss = 0.0
    all_hyps_phi, all_hyps_phibar, all_refs = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            frames = batch["frames"].to(device).permute(0, 3, 1, 2)
            frame_lengths = batch["frame_lengths"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            outputs = model(frames, frame_lengths, keep_prob=1.0)
            loss = compute_cosign_loss(
                outputs, targets, target_lengths, criterion, keep_prob=1.0)
            total_loss += loss.item()

            for branch, hyps_list in [("phi", all_hyps_phi), ("phi_inv", all_hyps_phibar)]:
                logits = outputs[branch]["main_logits"]
                log_probs = F.log_softmax(logits, dim=-1).cpu()
                seq_lengths = outputs[branch]["logit_lengths"].cpu()
                decode_results = ctc_decoder_obj(log_probs, seq_lengths)
                for i in range(targets.size(0)):
                    if len(decode_results[i]) > 0:
                        hyp = decode_results[i][0].tokens.tolist()
                    else:
                        hyp = []

                    hyps_list.append([id2gloss.get(v, "")
                                     for v in hyp if v != 0])

                    if branch == "phi":
                        ref = [id2gloss.get(
                            v.item(), "") for v in targets[i][:target_lengths[i]] if v.item() != 0]
                        all_refs.append(ref)

    avg_wer = compute_wer(all_hyps_phi, all_refs)
    phibar_wer = compute_wer(all_hyps_phibar, all_refs)

    for i in range(min(3, len(all_refs))):
        print(f"TARGET   : {' '.join(all_refs[i])}")
        print(f"PHI PRED : {' '.join(all_hyps_phi[i])}")
        print()

    return total_loss / len(dataloader), avg_wer, avg_wer, phibar_wer


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("Building vocabulary...")
    gloss2id, id2gloss = build_gloss_vocab([ANN_TRAIN, ANN_DEV])
    num_classes = len(gloss2id)

    print("Loading full datasets...")
    train_dataset = PhoenixDataset(DATA_DIR_TRAIN, ANN_TRAIN, gloss2id)
    dev_dataset = PhoenixDataset(DATA_DIR_DEV, ANN_DEV, gloss2id)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=phoenix_ctc_collate_fn, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=phoenix_ctc_collate_fn, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    print("Setting up CTC decoder...")
    tokens = [id2gloss[i] for i in range(num_classes)]
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write('\n'.join(tokens) + '\n')
        token_file = f.name

    ctc_decoder_obj = decoder.ctc_decoder(
        lexicon=None, tokens=token_file, blank_token=id2gloss[0], sil_token=id2gloss[0]
    )

    print(f"Initializing model on {DEVICE}...")
    model = CoSign1SModel(num_classes=num_classes, dropout=MODEL_DROPOUT)
    model.to(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=OPTIMIZER_MILESTONES, gamma=OPTIMIZER_GAMMA)
    criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    start_epoch = 0
    latest_ckpt = os.path.join(CHECKPOINT_DIR, "latest.pth")
    if os.path.exists(latest_ckpt):
        start_epoch, _, _ = load_checkpoint(latest_ckpt, model, optimizer)

    best_wer = float('inf')

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_train_loss = 0
        num_batches = 0

        # Annealing Cosine Scale (stops at 25.0)
        # current_scale = min(25.0, 1.0 + (epoch * 2.0))
        # model.gloss_head.scale = current_scale

        current_keep_prob = KEEP_PROB

        print(f"Epoch {
              epoch+1:3d} | Keep Prob: {current_keep_prob:.1f} | ")

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            frames = batch["frames"].to(DEVICE).permute(0, 3, 1, 2)
            frame_lengths = batch["frame_lengths"].to(DEVICE)
            targets = batch["targets"].to(DEVICE)
            target_lengths = batch["target_lengths"].to(DEVICE)

            outputs = model(frames, frame_lengths, keep_prob=current_keep_prob)
            loss = compute_cosign_loss(
                outputs, targets, target_lengths, criterion, kl_weight=KL_WEIGHT, keep_prob=current_keep_prob)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_train_loss += loss.item()
            num_batches += 1
            if (batch_idx) % 100 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / num_batches
        scheduler.step()

        # Eval on DEV set
        val_loss, avg_wer, phi_wer, phibar_wer = evaluate(
            model, dev_loader, criterion, ctc_decoder_obj, id2gloss, DEVICE
        )

        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"WER: {avg_wer:.3f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        save_checkpoint(model, optimizer, epoch + 1,
                        gloss2id, val_loss, latest_ckpt)
        if avg_wer < best_wer:
            best_wer = avg_wer
            best_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            save_checkpoint(model, optimizer, epoch + 1,
                            gloss2id, val_loss, best_path)
            print(f"-> NEW BEST WER: {best_wer:.3f}")


if __name__ == "__main__":
    main()
