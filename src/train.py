import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio.models.decoder as decoder
import tempfile
import json
import jiwer
import torch.nn.functional as F

from model import CoSign1SModel
from phoenix_dataloader import PhoenixDataset, phoenix_ctc_collate_fn, build_gloss_vocab

EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"

DATA_DIR_TRAIN = "pheonix-dataset/train"
DATA_DIR_DEV = "pheonix-dataset/dev"
ANN_TRAIN = "annotations/PHOENIX-2014-T.train.corpus.csv"
ANN_DEV = "annotations/PHOENIX-2014-T.dev.corpus.csv"


def save_checkpoint(model, optimizer, epoch, gloss2id, val_loss, filepath):
    """Saves the training state to a file."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'gloss2id': gloss2id,
        'val_loss': val_loss
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer):
    """Loads the training state from a file."""
    checkpoint = torch.load(filepath, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_loss = checkpoint.get('val_loss', float('inf'))

    print(f"Loaded checkpoint '{filepath}' (Resuming from epoch {epoch})")
    return epoch, model, optimizer


def compute_wer(hypotheses, references):
    """
    Computes Word Error Rate using jiwer.
    Expects lists of lists of words: [['hello', 'world'], ...]
    """
    hyp_strs = [" ".join(h) if len(h) > 0 else "<empty>" for h in hypotheses]
    ref_strs = [" ".join(r) if len(r) > 0 else "<empty>" for r in references]

    return jiwer.wer(ref_strs, hyp_strs)


def compute_cosign_loss(outputs, targets, target_lengths, criterion, kl_weight=0.1):
    """CoSign loss: CTC on all heads + symmetric KL between complementary branches."""
    total_loss = 0.0

    # 1. CTC losses for all 4 prediction heads (2 branches x 2 heads each)
    ctc_losses = []
    logit_lengths_dict = {}

    for branch in ["phi", "phi_inv"]:
        branch_out = outputs[branch]

        for head in ["aux_logits", "main_logits"]:
            logits = branch_out[head]  # [B, T', V]
            logit_lengths = branch_out["logit_lengths"]  # [B]

            log_probs = F.log_softmax(
                logits, dim=-1).transpose(0, 1)  # [T', B, V]
            ctc_loss = criterion(log_probs, targets,
                                 logit_lengths, target_lengths)
            ctc_losses.append(ctc_loss)
            logit_lengths_dict[f"{branch}_{head}"] = logit_lengths

    ctc_loss = torch.stack(ctc_losses).mean()  # Average 4 CTC losses
    total_loss += ctc_loss

    # 2. Complementary regularization (symmetric KL divergence)
    # Between auxiliary predictions of phi vs phi_inv
    aux_phi = F.log_softmax(outputs["phi"]["aux_logits"], dim=-1)
    aux_phibar = F.log_softmax(outputs["phi_inv"]["aux_logits"], dim=-1)
    aux_phi_soft = F.softmax(aux_phi, dim=-1)
    aux_phibar_soft = F.softmax(aux_phibar, dim=-1)

    kl_aux = F.kl_div(aux_phi, aux_phibar_soft, reduction='batchmean') + \
        F.kl_div(aux_phibar, aux_phi_soft, reduction='batchmean')

    # Between main predictions of phi vs phi_inv
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
    total_loss = 0
    all_hyps_phi = []
    all_hyps_phibar = []
    all_refs = []

    with torch.no_grad():
        for batch in dataloader:
            frames = batch["frames"].to(device)
            frames = frames.permute(0, 3, 1, 2)  # [B, 3, T, V]
            frame_lengths = batch["frame_lengths"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            outputs = model(frames, frame_lengths)
            loss = compute_cosign_loss(
                outputs, targets, target_lengths, criterion)
            total_loss += loss.item()

            for branch_name in ["phi", "phi_inv"]:
                logits = outputs[branch_name]["main_logits"]
                log_probs = F.log_softmax(logits, dim=-1)  # [B, T', V]
                emissions = log_probs.cpu()
                seq_lengths = outputs[branch_name]["logit_lengths"].cpu()

                decode_results = ctc_decoder_obj(emissions, seq_lengths)

                for i, result in enumerate(decode_results):
                    best_hyp = result[0].tokens.tolist()
                    hyp_words = [id2gloss.get(v, "<unk>")
                                 for v in best_hyp if v != 0]

                    if branch_name == "phi":
                        all_hyps_phi.append(hyp_words)
                    else:
                        all_hyps_phibar.append(hyp_words)

                    # Reference (once per batch item)
                    ref_words = [id2gloss.get(
                        v.item(), "<unk>") for v in targets[i][:target_lengths[i]] if v.item() != 0]
                    all_refs.append(ref_words)

    phi_wer = compute_wer(all_hyps_phi, all_refs[:len(all_hyps_phi)])
    phibar_wer = compute_wer(all_hyps_phibar, all_refs[:len(all_hyps_phibar)])
    avg_wer = (phi_wer + phibar_wer) / 2
    return total_loss / len(dataloader), avg_wer, phi_wer, phibar_wer


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("Building vocabulary")
    gloss2id, id2gloss = build_gloss_vocab([ANN_TRAIN, ANN_DEV])
    num_classes = len(gloss2id)
    print(f"Vocabulary size: {num_classes}")

    print("Loading datasets")
    train_dataset = PhoenixDataset(DATA_DIR_TRAIN, ANN_TRAIN, gloss2id)
    dev_dataset = PhoenixDataset(DATA_DIR_DEV, ANN_DEV, gloss2id)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=phoenix_ctc_collate_fn, num_workers=2, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=phoenix_ctc_collate_fn, num_workers=2, pin_memory=True)

    print("Setting up CTC decoder...")
    tokens = [id2gloss[i] for i in range(num_classes)]

    # torchaudio expects ONE TOKEN PER LINE
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write('\n'.join(tokens) + '\n')
        token_file = f.name

    ctc_decoder_obj = decoder.ctc_decoder(
        lexicon=None,
        tokens=token_file,
        blank_token=id2gloss[0],
        sil_token=id2gloss[0],
    )

    print(f"Initializing model on {DEVICE}...")
    model = CoSign1SModel(num_classes=num_classes)
    model.to(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    start_epoch = 0
    latest_ckpt = os.path.join(CHECKPOINT_DIR, "latest.pth")
    if os.path.exists(latest_ckpt):
        start_epoch, _, _ = load_checkpoint(latest_ckpt, model, optimizer)

    best_wer = float('inf')
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_train_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            frames = batch["frames"].to(DEVICE)
            frames = frames.permute(0, 3, 1, 2)
            frame_lengths = batch["frame_lengths"].to(DEVICE)
            targets = batch["targets"].to(DEVICE)
            target_lengths = batch["target_lengths"].to(DEVICE)

            outputs = model(frames, frame_lengths)
            loss = compute_cosign_loss(
                outputs, targets, target_lengths, criterion)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_train_loss += loss.item()
            num_batches += 1
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch {epoch+1:3d} | Batch {batch_idx +
                      1:4d}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / num_batches
        scheduler.step(avg_train_loss)

        val_loss, avg_wer, phi_wer, phibar_wer = evaluate(
            model, dev_loader, criterion, ctc_decoder_obj, id2gloss, DEVICE
        )

        print(f"Epoch {epoch+1:3d} | Train: {avg_train_loss:.4f} | Val: {val_loss:.4f} "
              f"| WER: {avg_wer:.3f} (phi:{phi_wer:.3f}, phi_inv:{
            phibar_wer:.3f}) "
            f"| LR: {scheduler.get_last_lr()[0]:.2e}")

        save_checkpoint(model, optimizer, epoch + 1,
                        gloss2id, val_loss, latest_ckpt)
        if avg_wer < best_wer:
            best_wer = avg_wer
            best_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            save_checkpoint(model, optimizer, epoch + 1,
                            gloss2id, val_loss, best_path)
            print(f"NEW BEST: {best_wer:.3f}")


if __name__ == "__main__":
    main()
