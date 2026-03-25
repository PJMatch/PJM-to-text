import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchaudio.models.decoder as decoder
import tempfile
import jiwer
import yaml
from tqdm import tqdm

from model import CoSign1SModel
from phoenix_dataloader import PhoenixDataset, phoenix_ctc_collate_fn, build_gloss_vocab


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_wer(hypotheses, references):
    hyp_strs = [" ".join(h) if len(h) > 0 else "<empty>" for h in hypotheses]
    ref_strs = [" ".join(r) if len(r) > 0 else "<empty>" for r in references]
    return jiwer.wer(ref_strs, hyp_strs)


def main():
    config = load_config()

    if config["system"]["device"] == "auto":
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        DEVICE = torch.device(config["system"]["device"])

    CHECKPOINT_DIR = config["system"]["checkpoint_dir"]
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    LM_MODEL = config["training"]["lm_model"]
    LEXICON = config["training"]["lexicon"]

    print("Loading vocabulary...")
    gloss2id, id2gloss = build_gloss_vocab(
        [config["data"]["train_ann"], config["data"]["dev_ann"]])
    num_classes = len(gloss2id)

    print("Loading Dev dataset...")
    dev_dataset = PhoenixDataset(
        config["data"]["dev_dir"], config["data"]["dev_ann"], gloss2id)
    dev_loader = DataLoader(dev_dataset, batch_size=config["training"]["batch_size"], shuffle=False,
                            collate_fn=phoenix_ctc_collate_fn, num_workers=config["data"]["num_workers"])

    print(f"Loading best model weights from {BEST_MODEL_PATH}...")
    model = CoSign1SModel(num_classes=num_classes,
                          dropout=config["model"]["dropout"])
    model.to(DEVICE)

    checkpoint = torch.load(
        BEST_MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("Pre-computing acoustic log-probabilities on GPU...")
    all_log_probs = []
    all_seq_lengths = []
    all_refs = []

    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="GPU Pre-computation"):
            frames = batch["frames"].to(DEVICE).permute(0, 3, 1, 2)
            frame_lengths = batch["frame_lengths"].to(DEVICE)
            targets = batch["targets"]
            target_lengths = batch["target_lengths"]

            outputs = model(frames, frame_lengths, keep_prob=1.0)
            logits = outputs["phi"]["main_logits"]

            log_probs = F.log_softmax(logits, dim=-1).cpu()
            all_log_probs.append(log_probs)
            all_seq_lengths.append(outputs["phi"]["logit_lengths"].cpu())

            for i in range(targets.size(0)):
                ref = [id2gloss.get(v.item(), "") for v in targets[i]
                                                    [:target_lengths[i]] if v.item() != 0]
                all_refs.append(ref)

    tokens = [id2gloss[i] for i in range(num_classes)]
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write('\n'.join(tokens) + '\n')
        token_file = f.name

    lm_weights = [0.5, 1.0, 1.5, 2.0, 2.5]
    word_scores = [-2.0, -1.0, 0.0, 1.0, 2.0]

    BEAM_SIZE = 250
    BEAM_THRESHOLD = 50.0

    total_combinations = len(lm_weights) * len(word_scores)

    print("\n" + "="*50)
    print(f"STARTING GRID SEARCH (Beam Size: {BEAM_SIZE} | Combinations: {total_combinations})")
    print("="*50)

    best_wer = float('inf')
    best_params = {}
    results_log = []

    current_combo = 1
    for lm_wt in lm_weights:
        for w_score in word_scores:

            ctc_decoder_obj = decoder.ctc_decoder(
                lexicon=LEXICON,
                tokens=token_file,
                blank_token=id2gloss[0],
                sil_token=id2gloss[0],
                beam_size=BEAM_SIZE,
                beam_threshold=BEAM_THRESHOLD,
                lm=LM_MODEL,
                lm_weight=lm_wt,
                word_score=w_score
            )

            all_hyps = []

            desc_str = f"Combo {current_combo}/{total_combinations} [LM:{lm_wt} WS:{w_score}]"
            for log_probs, seq_lengths in tqdm(zip(all_log_probs, all_seq_lengths), total=len(all_log_probs), desc=desc_str, leave=False):
                decode_results = ctc_decoder_obj(log_probs, seq_lengths)

                for i in range(log_probs.size(0)):
                    if len(decode_results[i]) > 0:
                        hyp = decode_results[i][0].tokens.tolist()
                    else:
                        hyp = []
                    all_hyps.append([id2gloss.get(v, "")
                                    for v in hyp if v != 0])

            current_wer = compute_wer(all_hyps, all_refs)
            print(f"[{current_combo}/{total_combinations}] LM Weight: {lm_wt:4.1f} | Word Score: {w_score:4.1f} | WER: {current_wer:.4f}")

            results_log.append((lm_wt, w_score, current_wer))

            if current_wer < best_wer:
                best_wer = current_wer
                best_params = {"lm_weight": lm_wt, "word_score": w_score}
            current_combo += 1

    print("\n" + "="*50)
    print("GRID SEARCH COMPLETE!")
    print(f"Best WER: {best_wer:.4f}")
    print(f"Best Parameters: LM Weight = {best_params['lm_weight']}, Word Score = {best_params['word_score']}")
    print("="*50)

if __name__ == "__main__":
    main()
