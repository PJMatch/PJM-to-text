"""Module for the PJM predictor."""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_path not in sys.path:
    sys.path.append(src_path)
from model import CoSign1SModel  # noqa: E402
from pjm_dataloader import build_gloss_vocab as build_pjm_vocab  # noqa: E402

#TODO: tweak this value
CONFIDENCE_THRESHOLD = 0.5


class SentenceSmoother:
    def __init__(self, similarity_threshold=0.3):
        self.similarity_threshold = similarity_threshold
        self.current_cluster = []
        self.last_emitted_sentence = ""

    def process(self, raw_text):
        """Takes the raw output from the NN every 15 frames.

        Returns a clean sentence only when the user finishes a thought,
        otherwise returns None.
        """
        if not raw_text or raw_text == "none" or len(raw_text.split()) < 2:
            return self._commit()

        words = raw_text.split()

        if not self.current_cluster:
            self.current_cluster.append(words)
            return None

        last_words = self.current_cluster[-1]

        intersection = len(set(words) & set(last_words))
        union = len(set(words) | set(last_words))
        similarity = intersection / union if union > 0 else 0

        if similarity > self.similarity_threshold:
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
        T = Time (frames), V = Vertices, C = Channels (x, y, z)
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

        predicted_text = self._greedy_decode_single(logits[0], logit_lengths[0].item())

        return predicted_text

    def _greedy_decode_single(self, sequence_logits, length, blank=0):
        """CTC greedy decoding with per-gloss confidence filtering.
        Sequence_logits shape: (T, num_classes)
        Glosses whose average softmax confidence is below CONFIDENCE_THRESHOLD
        are silently dropped from the output.
        """
        logits = sequence_logits[:length]
        probs = F.softmax(logits, dim=-1)
        max_probs, argmax = probs.max(dim=-1)

        result = []  # list of gloss_id,confidences
        prev = blank
        for t in range(len(argmax)):
            token = argmax[t].item()
            if token != blank and token != prev:
                result.append((token, [max_probs[t].item()]))
            elif token != blank and token == prev:
                result[-1][1].append(max_probs[t].item())
            prev = token

        hyp_words = []
        for gloss_id, scores in result:
            avg_conf = sum(scores) / len(scores)
            if avg_conf >= CONFIDENCE_THRESHOLD:
                hyp_words.append(self.id2gloss.get(gloss_id, "<unk>"))

        return " ".join(hyp_words)
