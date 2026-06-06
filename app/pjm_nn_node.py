"""Module for the PJM predictor."""

import os
import re
import sys
from dataclasses import dataclass

import consts
import numpy as np
import torch
import torch.nn.functional as F
import yaml

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_path not in sys.path:
    sys.path.append(src_path)
from model import CoSign1SModel


@dataclass
class GlossPrediction:
    """A single decoded gloss with its confidence."""

    name: str
    confidence: float


class GlossTracker:
    """Accumulates gloss predictions across sliding windows to filter noise."""

    def __init__(self, max_history=15):
        self.max_history = max_history
        self.predictions: list[list[GlossPrediction]] = []

    def vote(self, batch: list[GlossPrediction]) -> str | None:
        """
        Adds the batch to history and returns the currently highest-voted string.
        """
        if not batch:
            return None

        self.predictions.append(batch)
        if len(self.predictions) > self.max_history:
            self.predictions = self.predictions[-self.max_history :]

        return self._resolve_votes()

    def _resolve_votes(self) -> str | None:
        """Resolve glosses by name voting across windows."""
        if not self.predictions:
            return None

        ref_batch = self.predictions[-1]
        if not ref_batch:
            return None

        output_parts = []
        for ref_gloss in ref_batch:
            votes = 0
            conf_sum = 0.0

            for batch in self.predictions:
                match = next((g for g in batch if g.name == ref_gloss.name), None)
                if match:
                    votes += 1
                    conf_sum += match.confidence

            if votes >= consts.VOTE_THRESHOLD:
                avg_conf = conf_sum / votes
                if avg_conf >= 0.25:
                    output_parts.append(ref_gloss.name)

        return " ".join(output_parts) if output_parts else None


class SentenceSmoother:
    """Smoother logic for parsing raw NN output into human-readable sentences."""

    def __init__(self, similarity_threshold=0.2):
        self.similarity_threshold = similarity_threshold
        self.current_cluster = []
        self.last_emitted_sentence = ""

    def process(self, raw_text: str) -> str | None:
        """
        Detects when the user transitions to a new thought by measuring
        similarity against the PEAK of the current thought.
        """
        if not raw_text:
            return None

        clean_text = re.sub(r"\(votes=\d+ conf=[0-9.]+\)", "", raw_text).strip()
        words = clean_text.split()

        if not words:
            return None

        if not self.current_cluster:
            self.current_cluster.append(words)
            return None

        peak_words = max(self.current_cluster, key=len)

        intersection = len(set(words) & set(peak_words))
        union = len(set(words) | set(peak_words))
        similarity = intersection / union if union > 0 else 0

        if similarity >= self.similarity_threshold or set(words).issubset(set(peak_words)):
            self.current_cluster.append(words)
            return None

        else:
            clean_sentence = self._commit()
            self.current_cluster.append(words)
            return clean_sentence

    def _commit(self) -> str | None:
        """Finds the longest sentence in the cluster and filters out noise."""
        if not self.current_cluster:
            return None

        best_words = max(self.current_cluster, key=len)
        final_sentence = " ".join(best_words)
        cluster_duration = len(self.current_cluster)

        self.current_cluster = []

        if len(best_words) < 2 and cluster_duration < 4:
            return None

        if cluster_duration < 3:
            return None

        if final_sentence and final_sentence != self.last_emitted_sentence:
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

        ckpt_path = os.path.join(self.config["system"]["checkpoint_dir"], "best_model.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        self.gloss2id = checkpoint["gloss2id"]
        self.id2gloss = {v: k for k, v in self.gloss2id.items()}
        num_classes = len(self.gloss2id)

        self.model = CoSign1SModel(num_classes=num_classes, dropout=0.0)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model weights from epoch {checkpoint.get('epoch', 'unknown')}")

        self.model.to(self.device)
        self.model.eval()

    def predict(self, window_chunk, window_start_frame=0) -> list[GlossPrediction]:
        """Runs inference on a single sliding window.

        Args:
            window_chunk: array of shape (T, V, C)
            window_start_frame: absolute frame index (unused, kept for interface compat)

        Returns:
            list of GlossPrediction with confidence scores.
            Empty list means silence / no glosses detected.
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

        raw = self._greedy_decode_single(logits[0], logit_lengths[0].item())

        predictions = []
        for gloss_id, _, _, conf in raw:
            name = self.id2gloss.get(gloss_id, "<unk>")
            predictions.append(GlossPrediction(name=name, confidence=conf))

        return predictions

    def _greedy_decode_single(self, sequence_logits, length, blank=0):
        """CTC greedy decoding that tracks temporal positions.

        Returns:
            list of (gloss_id, logit_start, logit_end, avg_confidence) tuples.
            logit_start/end are positions in the logit sequence (before upsampling).
        """
        valid_logits = sequence_logits[:length]
        probs = F.softmax(valid_logits, dim=-1)
        max_probs, preds = torch.max(probs, dim=-1)

        result = []
        prev_token = -1

        for t in range(len(preds)):
            token = preds[t].item()
            if token != blank and token != prev_token:
                result.append((token, t, t, [max_probs[t].item()]))
            elif token != blank and token == prev_token:
                _, start, _, scores = result[-1]
                result[-1] = (token, start, t, scores + [max_probs[t].item()])
            prev_token = token

        return [
            (gloss_id, start, end, sum(scores) / len(scores))
            for gloss_id, start, end, scores in result
        ]
