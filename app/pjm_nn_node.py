"""
Module handling the core neural network prediction logic and text post-processing.
"""

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
    """
    Smoothing logic for parsing raw neural network outputs into human-readable, stable sentences.
    Merges overlapping sliding window predictions.
    """
    
    def __init__(self, similarity_threshold: float = 0.5, patience: int = 3) -> None:
        """
        Initializes the sentence smoother mechanism.

        Args:
            similarity_threshold (float): Minimum overlap required to cluster words. Defaults to 0.5.
            patience (int): Number of empty predictions required to finalize a sentence. Defaults to 3.
        """
        self.similarity_threshold = similarity_threshold
        self.patience = patience
        self.current_cluster = []
        self.last_emitted_sentence = ""
        self.none_counter = 0

    def process(self, raw_text: str) -> str | None:
        """
        Processes raw text output from the neural network at each stride.

        Args:
            raw_text (str): The raw decoded sequence from the model.

        Returns:
            str | None: A finalized clean sentence if silence is detected, otherwise None.
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

    def _commit(self) -> str | None:
        """
        Finalizes the active text cluster by extracting the longest sentence variation.

        Returns:
            str | None: The longest merged sentence string, or None if the cluster is empty.
        """
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
    """
    Wrapper class for the PyTorch sign language classification model.
    Handles device mapping, checkpoint loading, and CTC greedy decoding.
    """

    def __init__(self, config_path: str = "config.yaml", mode: str = "CSLR") -> None:
        """
        Initializes the predictor, loads vocabularies and pre-trained weights.

        Args:
            config_path (str): Path to the YAML configuration file. Defaults to "config.yaml".
            mode (str): Application mode ('CSLR' or 'ISLR'). Defaults to "CSLR".
        """
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

    def predict(self, window_chunk: list) -> str:
        """
        Feeds a single window of spatial landmarks into the neural network.

        Args:
            window_chunk (list): A list representing the sliding window frame data.

        Returns:
            str: The decoded sentence or word prediction.
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

    def _greedy_decode_single(self, sequence_logits: torch.Tensor, length: int, blank: int = 0, threshold: float = 0.6) -> str:
        """
        Applies standard CTC greedy decoding with a confidence threshold filter.

        If the model's highest probability for a class is below the threshold, 
        it forces the output to be interpreted as the blank token (silence/noise).

        Args:
            sequence_logits (torch.Tensor): Raw model output logits.
            length (int): Valid temporal length of the sequence.
            blank (int): Index of the CTC blank token. Defaults to 0.
            threshold (float): Minimum confidence required to accept a non-blank token. Defaults to 0.6.

        Returns:
            str: A string of space-separated gloss translations.
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