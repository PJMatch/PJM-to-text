"""Module for the PJM predictor."""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_path not in sys.path:
    sys.path.append(src_path)
from model import CoSign1SModel  # noqa: E402
from pjm_dataloader import build_gloss_vocab as build_pjm_vocab  # noqa: E402


class SentenceSmoother:
    """Smoother logic for parsing raw NN output into human-readable sentences."""
    
    def __init__(self, similarity_threshold=0.5, patience=3):
        self.similarity_threshold = similarity_threshold
        self.patience = patience
        self.current_cluster = []
        self.last_emitted_sentence = ""
        self.none_counter = 0

    def process(self, raw_text):
        """Takes the raw output from the NN every stride frames.

        Returns a clean sentence only when the user finishes a thought (silence detected),
        otherwise returns None.
        """
        text_lower = str(raw_text).strip().lower()

        if not text_lower or text_lower == "none":
            self.none_counter += 1
            if self.none_counter >= self.patience:
                return self._commit()
            return None

        self.none_counter = 0

        words = raw_text.split()
        if not self.current_cluster and len(words) < 2:
            self.current_cluster.append(words)
            return None

        if not self.current_cluster:
            self.current_cluster.append(words)
            return None

        last_words = self.current_cluster[-1]

        intersection = len(set(words) & set(last_words))
        union = len(set(words) | set(last_words))
        similarity = intersection / union if union > 0 else 0

        if similarity > self.similarity_threshold or set(last_words).issubset(set(words)):
            self.current_cluster.append(words)
            return None
        else:
            clean_sentence = self._commit()
            self.current_cluster.append(words)
            return clean_sentence

    def _commit(self):
        """Finds the longest sentence in the cluster and returns it."""
        if not self.current_cluster:
            return None

        best_words = max(self.current_cluster, key=len)
        final_sentence = " ".join(best_words)

        self.current_cluster = []
        self.none_counter = 0

        if final_sentence != self.last_emitted_sentence:
            self.last_emitted_sentence = final_sentence
            return final_sentence

        return None


class PJMPredictor:
    """PJM neural network predictor."""

    def __init__(self, config_path="config.yaml"):
        """Runs when the app starts. Loads config, vocab, model, and weights."""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device(
            self.config["system"]["device"]
            if self.config["system"]["device"] != "auto"
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        src_path = Path(__file__).parent.resolve() / "../src"

        annotation_dir = src_path / self.config["data"]["annotation_dir"]
        train_ann = src_path / self.config["data"]["train_ann"]
        test_ann = src_path / self.config["data"]["test_ann"]

        self.gloss2id, self.id2gloss = build_pjm_vocab(annotation_dir, [train_ann, test_ann])
        num_classes = len(self.gloss2id)

        self.model = CoSign1SModel(num_classes=num_classes, dropout=0.0)

        ckpt_path = os.path.join(self.config["system"]["checkpoint_dir"], "best_model.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model weights from epoch {checkpoint.get('epoch', 'unknown')}")

        self.model.to(self.device)
        self.model.eval()

    def predict(self, window_chunk):
        """Runs continuously.

        Takes a single sliding window of landmarks and returns the predicted sentence.
        Expected window_chunk shape: (T, V, C)
        """
        window_chunk = np.array(window_chunk)
        frames_tensor = torch.tensor(window_chunk, dtype=torch.float32).unsqueeze(0).to(self.device)

        frames_tensor = frames_tensor.permute(0, 3, 1, 2)

        seq_len = frames_tensor.size(2)
        frame_lengths = torch.tensor([seq_len], dtype=torch.long).to(self.device)

        with torch.no_grad():
            outputs = self.model(frames_tensor, frame_lengths, keep_prob=1.0)
            logits = outputs["phi"]["main_logits"]
            logit_lengths = outputs["phi"]["logit_lengths"].cpu()

        predicted_text = self._greedy_decode_single(logits[0], logit_lengths[0].item(), threshold=0.6)

        return predicted_text

    def _greedy_decode_single(self, sequence_logits, length, blank=0, threshold=0.6):
        """Standard CTC greedy decoding for a single sequence with Logit Thresholding.
        
        If the model's confidence for the predicted class is below the threshold,
        it forces the output to be 'blank' (silence/none).
        """
        valid_logits = sequence_logits[:length]
        
        probs = torch.softmax(valid_logits, dim=-1)
        
        max_probs, preds = torch.max(probs, dim=-1)
        
        preds[max_probs < threshold] = blank

        hyp = []
        prev_token = -1

        for token_tensor in preds:
            token = token_tensor.item()
            if token != blank and token != prev_token:
                hyp.append(token)
            prev_token = token

        hyp_words = [self.id2gloss.get(v, "<unk>") for v in hyp]
        return " ".join(hyp_words)