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
from torch.utils.data import Subset

from model import CoSign1SModel
from phoenix_dataloader import PhoenixDataset, phoenix_ctc_collate_fn, build_gloss_vocab

EPOCHS = 1000
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"

DATA_DIR_TRAIN = "pheonix-dataset/train"
DATA_DIR_DEV = "pheonix-dataset/dev"
ANN_TRAIN = "annotations/PHOENIX-2014-T.train.corpus.csv"
ANN_DEV = "annotations/PHOENIX-2014-T.dev.corpus.csv"

def greedy_ctc_decode(logits, lengths, blank=0):
    pred = logits.argmax(dim=-1)  # [B, T]
    decoded = []

    for b in range(pred.size(0)):
        seq = pred[b, :lengths[b]].tolist()
        out = []
        prev = None
        for tok in seq:
            if tok != blank and tok != prev:
                out.append(tok)
            prev = tok
        decoded.append(out)

    return decoded, pred


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



def compute_cosign_loss(outputs, targets, target_lengths, criterion):
    logits = outputs["phi"]["main_logits"]
    logit_lengths = outputs["phi"]["logit_lengths"]

    if (logit_lengths < target_lengths).any():
        print("BAD LENGTHS!", logit_lengths, target_lengths)

    log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
    return criterion(log_probs, targets, logit_lengths, target_lengths)


def evaluate(model, dataloader, criterion, ctc_decoder_obj, id2gloss, device):
    model.eval()
    total_loss = 0.0
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

            outputs = model(frames, frame_lengths, keep_prob=1.0)
            loss = compute_cosign_loss(
                outputs, targets, target_lengths, criterion)
            total_loss += loss.item()

            logits_phi = outputs["phi"]["main_logits"]
            log_probs_phi = F.log_softmax(logits_phi, dim=-1).cpu()
            seq_lengths_phi = outputs["phi"]["logit_lengths"].cpu()
            decode_results_phi = ctc_decoder_obj(
                log_probs_phi, seq_lengths_phi)

            logits_phibar = outputs["phi_inv"]["main_logits"]
            log_probs_phibar = F.log_softmax(logits_phibar, dim=-1).cpu()
            seq_lengths_phibar = outputs["phi_inv"]["logit_lengths"].cpu()
            decode_results_phibar = ctc_decoder_obj(
                log_probs_phibar, seq_lengths_phibar)

            batch_size = targets.size(0)
            for i in range(batch_size):
                hyp_phi = decode_results_phi[i][0].tokens.tolist()
                hyp_phibar = decode_results_phibar[i][0].tokens.tolist()

                hyp_phi_words = [id2gloss.get(v, "")
                                 for v in hyp_phi if v != 0]
                hyp_phibar_words = [id2gloss.get(
                    v, "") for v in hyp_phibar if v != 0]
                ref_words = [
                    id2gloss.get(v.item(), "")
                    for v in targets[i][:target_lengths[i]]
                    if v.item() != 0
                ]

                all_hyps_phi.append(hyp_phi_words)
                all_hyps_phibar.append(hyp_phibar_words)
                all_refs.append(ref_words)

            break

    phi_wer = compute_wer(all_hyps_phi, all_refs)
    phibar_wer = compute_wer(all_hyps_phibar, all_refs)
    avg_wer = phi_wer

    for i in range(min(3, len(all_refs), len(all_hyps_phi), len(all_hyps_phibar))):
        print(f"TARGET   : {' '.join(all_refs[i])}")
        print(f"PHI PRED : {' '.join(all_hyps_phi[i])}")
        print(f"INV PRED : {' '.join(all_hyps_phibar[i])}")
        print()

    return total_loss, avg_wer, phi_wer, phibar_wer


def debug_first_batch(model, dataloader, id2gloss, device):
    model.train()
    batch = next(iter(dataloader))

    with torch.no_grad():
        frames = batch["frames"].to(device).permute(0, 3, 1, 2)
        frame_lengths = batch["frame_lengths"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        outputs = model(frames, frame_lengths, keep_prob=1.0)

        logits = outputs["phi"]["main_logits"]                  # [B, T', V]
        pred_ids = logits.argmax(dim=-1)                       # [B, T']
        blank_ratio = (pred_ids == 0).float().mean().item()

        print("blank_ratio:", blank_ratio)
        print("logit_lengths:", outputs["phi"]["logit_lengths"].tolist())
        print("target_lengths:", target_lengths.tolist())

        for i in range(min(3, logits.size(0))):
            hyp = [id2gloss[idx.item()]
                   for idx in pred_ids[i] if idx.item() != 0]
            ref = [id2gloss[idx.item()]
                   for idx in targets[i][:target_lengths[i]] if idx.item() != 0]
            print("REF :", " ".join(ref))
            print("ARGM:", " ".join(hyp))
            print()

        print("Debug info: [target_lengths, logit_lengths, targets]")
        print(target_lengths[:2])
        print(outputs["phi"]["logit_lengths"][:2])
        print(targets[:2])


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("Building vocabulary")
    gloss2id, id2gloss = build_gloss_vocab([ANN_TRAIN, ANN_DEV])
    num_classes = len(gloss2id)
    print(f"Vocabulary size: {num_classes}")

    print("Loading datasets")

    # dev_dataset = PhoenixDataset(DATA_DIR_DEV, ANN_DEV, gloss2id)
    #
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    #                           collate_fn=phoenix_ctc_collate_fn, num_workers=2, pin_memory=True)
    # dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False,
    #                         collate_fn=phoenix_ctc_collate_fn, num_workers=2, pin_memory=True)
    train_dataset_full = PhoenixDataset(DATA_DIR_TRAIN, ANN_TRAIN, gloss2id)
    
    tiny_indices = [0] 
    train_dataset = Subset(train_dataset_full, tiny_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1, 
        shuffle=False,
        collate_fn=phoenix_ctc_collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    dev_dataset = PhoenixDataset(DATA_DIR_DEV, ANN_DEV, gloss2id)

    print("Setting up CTC decoder...")
    tokens = [id2gloss[i] for i in range(num_classes)]

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
    model = CoSign1SModel(num_classes=num_classes, dropout=0.0)
    model.to(DEVICE)

    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.CTCLoss(blank=0,reduction="mean", zero_infinity=False)

    debug_first_batch(model, train_loader, id2gloss, DEVICE)
    print("End")

    start_epoch = 0
    latest_ckpt = os.path.join(CHECKPOINT_DIR, "latest.pth")
    if os.path.exists(latest_ckpt):
        start_epoch, _, _ = load_checkpoint(latest_ckpt, model, optimizer)

    best_wer = float('inf')
    WARMUP_EPOCHS = 10

    for epoch in range(start_epoch, EPOCHS):

        model.train()
        total_train_loss = 0
        num_batches = 0

        current_scale = min(25.0, 1.0 + (epoch * 2.0))
        model.gloss_head.scale = current_scale

        print(f"Epoch {epoch} | Cosine Scale: {current_scale}")

        # Warmup phase without masking
        current_keep_prob = 1.0  # if epoch < WARMUP_EPOCHS else 0.8

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            frames = batch["frames"].to(DEVICE)
            frames = frames.permute(0, 3, 1, 2)
            frame_lengths = batch["frame_lengths"].to(DEVICE)
            targets = batch["targets"].to(DEVICE)
            target_lengths = batch["target_lengths"].to(DEVICE)

            outputs = model(frames, frame_lengths, keep_prob=current_keep_prob)
            loss = compute_cosign_loss(
                outputs, targets, target_lengths, criterion)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_train_loss += loss.item()
            num_batches += 1
            if(batch_idx == 0):
                break
            # if (batch_idx + 1) % 100 == 0:
            #     print(f"Epoch {epoch+1:3d} | Batch {batch_idx +
            #           1:4d}/{len(train_loader)} | Loss: {loss.item():.4f}")
            break

        # avg_train_loss = total_train_loss / num_batches
        # scheduler.step(avg_train_loss)

        val_loss, avg_wer, phi_wer, phibar_wer = evaluate(
            model, train_loader, criterion, ctc_decoder_obj, id2gloss, DEVICE
        )

        print(f"Epoch {epoch+1:3d} | "
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
