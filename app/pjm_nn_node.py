"""Module for the PJM predictor."""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import consts

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_path not in sys.path:
    sys.path.append(src_path)
from model import CoSign1SModel
from pjm_dataloader import build_gloss_vocab as build_pjm_vocab


@dataclass
class GlossPrediction:
    """A single decoded gloss with its confidence."""
    name: str
    confidence: float


class GlossTracker:
    """Accumulates gloss predictions across sliding windows and resolves
    which glosses are real by checking name persistence across windows.
    """

    def __init__(self, patience=3, max_history=15):
        self.patience = patience
        self.max_history = max_history
        self.predictions: list[list[GlossPrediction]] = []
        self.silence_counter = 0
        self.last_emitted_sentence = ""

    def vote(self, batch: list[GlossPrediction]):
        """Add a new window's gloss predictions to the tracker."""
        if batch:
            self.silence_counter = 0
        self.predictions.append(batch)
        if len(self.predictions) > self.max_history:
            self.predictions = self.predictions[-self.max_history:]

    def notify_silence(self) -> str | None:
        """Called when the model outputs nothing. Returns a resolved sentence
        if patience is exceeded, otherwise None."""
        self.silence_counter += 1
        if self.silence_counter >= self.patience:
            return self._commit()
        return None

    def _commit(self) -> str | None:
        """Resolve accumulated predictions into a final sentence."""
        self.silence_counter = 0
        result = self._resolve_debug()
        self.predictions = []
        if result and result != self.last_emitted_sentence:
            self.last_emitted_sentence = result
            return result
        return None

    # how many positions a gloss can shift between windows and still match.
    POSITION_TOLERANCE = 0

    def _resolve_debug(self) -> str | None:
        """Resolve glosses by name + ordinal position voting across windows. """
        if not self.predictions:
            return None

        ref_batch = self.predictions[-1]
        if not ref_batch:
            return None

        tol = self.POSITION_TOLERANCE

        output_parts = []
        for ref_idx, ref_gloss in enumerate(ref_batch):
            votes = 0
            conf_sum = 0.0

            for batch in self.predictions:
                for cand_idx, cand_gloss in enumerate(batch):
                    if cand_gloss.name == ref_gloss.name and abs(cand_idx - ref_idx) <= tol:
                        votes += 1
                        conf_sum += cand_gloss.confidence
                        break

            if votes < consts.VOTE_THRESHOLD:
                continue

            avg_conf = conf_sum / votes if votes > 0 else 0.0
            output_parts.append(f"{ref_gloss.name}(votes={votes} conf={avg_conf:.2f})")

        return " ".join(output_parts) if output_parts else None

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

        result = []  # list of gloss_id start_t end_t confidences
        prev_token = -1

        for t in range(len(preds)):
            token = preds[t].item()
            if token != blank and token != prev_token:
                result.append((token, t, t, [max_probs[t].item()]))
            elif token != blank and token == prev_token:
                _, start, _, scores = result[-1]
                result[-1] = (token, start, t, scores + [max_probs[t].item()])
            prev_token = token

        # Average confidence over the duration of each gloss
        return [
            (gloss_id, start, end, sum(scores) / len(scores))
            for gloss_id, start, end, scores in result
        ]
