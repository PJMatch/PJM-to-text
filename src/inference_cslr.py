import os
import argparse
import torch
import yaml
import tempfile
import jiwer
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchaudio.models.decoder import ctc_decoder

from model import CoSign1SModel
from pjm_dataloader_cslr import PJMDataset, pjm_ctc_collate_fn


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def greedy_decode(logits, seq_lengths, id2gloss, blank=0):
    """Fallback greedy CTC decode."""
    preds = torch.argmax(logits, dim=-1)  # [B, T]
    batch_hyps = []

    for i in range(preds.size(0)):
        hyp = []
        prev_token = -1
        for t in range(seq_lengths[i]):
            token = preds[i, t].item()
            if token != blank and token != prev_token:
                hyp.append(token)
            prev_token = token

        hyp_words = [id2gloss.get(v, "<unk>") for v in hyp]
        batch_hyps.append(" ".join(hyp_words))

    return batch_hyps


def build_lm_decoder(id2gloss, config, args):
    """Builds a torchaudio CTC beam search decoder powered by KenLM."""
    lm_model_path = config["training"].get("lm_model", None)
    lexicon_path = config["training"].get("lexicon", None)

    tokens = [id2gloss[i] for i in range(len(id2gloss))]
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("\n".join(tokens) + "\n")
        token_file = f.name

    beam_size = config["training"].get("beam_size", 50)

    lm_weight = (
        args.lm_weight if args.lm_weight is not None else config["training"].get("lm_weight", 1.5)
    )
    word_score = (
        args.word_score
        if args.word_score is not None
        else config["training"].get("word_score", 1.5)
    )

    print(
        f"Decoder Config -> LM Weight: {lm_weight} | Word Score: {word_score} | Beam: {beam_size}"
    )

    decoder_obj = ctc_decoder(
        lexicon=lexicon_path,
        tokens=token_file,
        lm=lm_model_path,
        nbest=1,
        beam_size=beam_size,
        lm_weight=lm_weight,
        word_score=word_score,
        blank_token="<blank>",
        sil_token="<blank>",
    )

    return decoder_obj


def main():
    parser = argparse.ArgumentParser(description="Inference for CoSign Model")
    parser.add_argument(
        "--greedy", action="store_true", help="Force greedy decoding instead of beam search"
    )
    parser.add_argument("--lm_weight", type=float, default=None, help="Override LM weight")
    parser.add_argument(
        "--word_score", type=float, default=None, help="Override Word Score (Insertion bonus)"
    )
    args = parser.parse_args()

    config = load_config()
    device = torch.device(
        config["system"]["device"]
        if config["system"]["device"] != "auto"
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Loading checkpoint")
    ckpt_path = os.path.join(config["system"]["checkpoint_dir"], "best_model.pth")
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    else:
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")

    gloss2id = checkpoint["gloss2id"]
    id2gloss = {v: k for k, v in gloss2id.items()}
    num_classes = len(gloss2id)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, num_classes={num_classes}")

    annotation_dir = config["data"]["annotation_dir"]
    print("Loading PJM dev dataset")
    test_dataset = PJMDataset(
        data_dir=config["data"]["dev_dir"],
        annotation_dir=annotation_dir,
        split_file=config["data"]["dev_ann"],
        gloss2id=gloss2id,
        split="test",
        use_temporal_aug=False,
    )
    collate_fn = pjm_ctc_collate_fn

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config["data"]["num_workers"],
    )

    if not args.greedy:
        print("Setting up LM Decoder...")
        decoder_obj = build_lm_decoder(id2gloss, config, args)
    else:
        print("Using purely GREEDY decoding (Argmax).")

    print("Loading Model")
    model = CoSign1SModel(num_classes=num_classes, dropout=0.0)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_hyps = []
    all_refs = []

    print("\n" + "=" * 50)
    print("STARTING INFERENCE")
    print("=" * 50)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            frames = batch["frames"].to(device).permute(0, 3, 1, 2)
            frame_lengths = batch["frame_lengths"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            outputs = model(frames, frame_lengths, keep_prob=1.0)

            logits = outputs["phi"]["main_logits"]
            seq_lengths = outputs["phi"]["logit_lengths"].cpu()

            if args.greedy:
                batch_hyps = greedy_decode(logits, seq_lengths, id2gloss, blank=0)
            else:
                log_probs = F.log_softmax(logits, dim=-1).cpu()
                decode_results = decoder_obj(log_probs, seq_lengths)

                batch_hyps = []
                for i in range(targets.size(0)):
                    best_hyp_ids = decode_results[i][0].tokens.tolist()
                    hyp_words = [id2gloss.get(v, "<unk>") for v in best_hyp_ids if v != 0]
                    batch_hyps.append(" ".join(hyp_words))

            for i in range(targets.size(0)):
                ref_ids = targets[i][: target_lengths[i]]
                ref_words = [id2gloss.get(v.item(), "<unk>") for v in ref_ids if v.item() != 0]
                ref_str = " ".join(ref_words)
                hyp_str = batch_hyps[i]

                all_hyps.append(hyp_str)
                all_refs.append(ref_str)

                print(f"Sample {len(all_refs)}:")
                print(f"REF : {ref_str}")
                print(f"HYP : {hyp_str}")
                print("-" * 50)


    out = jiwer.process_words(all_refs, all_hyps)

    print(f"Word Error Rate (WER) : {out.wer * 100:.2f}%")
    print(f"Match Error Rate (MER): {out.mer * 100:.2f}%")
    print(f"Word Info Lost (WIL)  : {out.wil * 100:.2f}%")

    print("\n--- Error Breakdown ---")
    total_words = sum(len(r.split()) for r in all_refs)
    print(f"Total Reference Words : {total_words}")
    print(
        f"Substitutions         : {out.substitutions} ({
            (out.substitutions / total_words) * 100:.2f}%)"
    )
    print(f"Deletions             : {out.deletions} ({(out.deletions / total_words) * 100:.2f}%)")
    print(f"Insertions            : {out.insertions} ({(out.insertions / total_words) * 100:.2f}%)")
    print(f"Hits (Correct)        : {out.hits} ({(out.hits / total_words) * 100:.2f}%)")


if __name__ == "__main__":
    main()
