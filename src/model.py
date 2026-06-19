"""
Core neural network architecture.

This module houses the core deep learning components, primarily focusing on Spatio-Temporal 
Graph Convolutional Networks (ST-GCN) to extract spatial relationships from skeletal graphs, 
and temporal modules (1D CNNs, BiLSTMs) to capture sequential dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from stgcn.stgcn import STGCNCoSign1s


class CoSignTemporalCNN(nn.Module):
    """
    Temporal convolutional network to extract features over time using 1D CNNs (C3-P2-C3-P2).
    Reduces the temporal resolution while increasing the feature dimensions.
    """

    def __init__(self, in_dim: int = 1024, hidden_dim: int = 1024, dropout: float = 0.2):
        """
        Initializes the Temporal CNN layers.

        Args:
            in_dim (int): The number of input channels/features. Defaults to 1024.
            hidden_dim (int): The number of hidden channels/features. Defaults to 1024.
            dropout (float): Dropout probability for regularization. Defaults to 0.2.
        """
        super().__init__()

        self.conv1 = nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, hidden_dim)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, hidden_dim)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _pool_out_lengths(lengths: torch.Tensor, kernel_size: int = 2, stride: int = 2, padding: int = 0, dilation: int = 1) -> torch.Tensor:
        """
        Calculates the new sequence length after a max pooling operation.

        Args:
            lengths (torch.Tensor): 1D tensor of original sequence lengths.
            kernel_size (int): Size of the pooling window. Defaults to 2.
            stride (int): Stride of the pooling window. Defaults to 2.
            padding (int): Padding added to the input. Defaults to 0.
            dilation (int): Dilation factor of the pooling. Defaults to 1.

        Returns:
            torch.Tensor: 1D tensor of updated, pooled sequence lengths.
        """
        out_lengths = (
            torch.div(
                lengths + 2 * padding - dilation * (kernel_size - 1) - 1,
                stride,
                rounding_mode="floor",
            )
            + 1
        )
        return torch.clamp(out_lengths, min=1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> tuple:
        """
        Performs the forward pass for temporal convolution and pooling.

        Args:
            x (torch.Tensor): Input feature tensor of shape [Batch, Channels, Time].
            lengths (torch.Tensor, optional): 1D tensor of sequence lengths. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - Output feature tensor of shape [Batch, Channels, PooledTime].
                - Updated 1D tensor of sequence lengths (if lengths were provided).
        """
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.pool2(x)

        if lengths is not None:
            lengths = self._pool_out_lengths(lengths)
            lengths = self._pool_out_lengths(lengths)

        return x, lengths


class LSTM(nn.Module):
    """
    Bidirectional LSTM to capture sequential context from temporal features.
    """

    def __init__(self, input_dim: int = 1024, hidden_size: int = 512, num_layers: int = 2, dropout: float = 0.2):
        """
        Initializes the BiLSTM layer.

        Args:
            input_dim (int): The number of input features per timestep. Defaults to 1024.
            hidden_size (int): The number of features in the hidden state. Defaults to 512.
            num_layers (int): Number of recurrent layers. Defaults to 2.
            dropout (float): Dropout probability between LSTM layers. Defaults to 0.2.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Performs the forward pass through the BiLSTM, utilizing sequence packing if lengths are provided.

        Args:
            x (torch.Tensor): Input sequence tensor of shape [Batch, Channels, Time].
            lengths (torch.Tensor, optional): 1D tensor of valid sequence lengths. Defaults to None.

        Returns:
            torch.Tensor: The output sequence tensor of shape [Batch, Time, 2 * HiddenSize].
        """
        x = x.transpose(1, 2)

        if lengths is None:
            out, _ = self.lstm(x)
            return out

        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        return out


class SharedGlossHead(nn.Module):
    """
    Classification head that uses cosine similarity instead of standard linear projection.
    """
    
    def __init__(self, feat_dim: int, vocab_size: int):
        """
        Initializes the Shared Gloss Head.

        Args:
            feat_dim (int): The dimensionality of the input features.
            vocab_size (int): The total number of classes/glosses in the vocabulary.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, feat_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = nn.Parameter(torch.tensor(25.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the scaled cosine similarity between input features and learned class weights.

        Args:
            x (torch.Tensor): Input feature tensor.

        Returns:
            torch.Tensor: The output logits/similarities of shape [..., VocabSize].
        """
        x_norm = F.normalize(x, dim=-1)
        w_norm = F.normalize(self.weight, dim=-1)
        sim = torch.matmul(x_norm, w_norm.t())
        return sim * self.scale


class CoSign1SModel(nn.Module):
    """
    One-stream CoSign model with complementary masking.
    
    Forward Pass Pipeline:
    1. Spatial Extraction: Skeleton sequences pass through the ST-GCN backbone.
    2. Branching & Masking: The feature map is duplicated. Branch 1 is multiplied by an attention mask (Phi), 
       and Branch 2 is multiplied by its inverse (1 - Phi).
    3. Temporal Compression: 1D Temporal CNNs (C3-P2) reduce sequence length.
    4. Contextualization: BiLSTM captures global sentence context.
    5. Classification: Shared Gloss Heads output final CTC log-probabilities.
    """

    def __init__(
        self,
        num_classes: int,
        stgcn_config_path: str = None,
        feat_dim: int = 1024,
        lstm_hidden: int = 512,
        dropout: float = 0.2,
    ):
        """
        Initializes the complete CoSign architecture.

        Args:
            num_classes (int): Number of target gloss classes.
            stgcn_config_path (str, optional): Path to the ST-GCN configuration file. Defaults to None.
            feat_dim (int): Dimensionality of intermediate features. Defaults to 1024.
            lstm_hidden (int): Hidden size for the BiLSTM layer. Defaults to 512.
            dropout (float): Dropout probability across modules. Defaults to 0.2.
        """
        super().__init__()

        self.STGCN = STGCNCoSign1s(config_path=stgcn_config_path)

        self.temporal_cnn = CoSignTemporalCNN(
            in_dim=feat_dim,
            hidden_dim=feat_dim,
            dropout=dropout,
        )

        self.context_lstm = LSTM(
            input_dim=feat_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            dropout=dropout,
        )

        self.gloss_head = SharedGlossHead(
            feat_dim=2 * lstm_hidden,
            vocab_size=num_classes,
        )

    def _forward_branch(self, x_branch: torch.Tensor, lengths: torch.Tensor) -> dict:
        """
        Processes a single complementary branch (either phi or 1-phi).

        Args:
            x_branch (torch.Tensor): Feature tensor for the specific branch of shape [Batch, Channels, Time].
            lengths (torch.Tensor): 1D tensor of unpadded sequence lengths.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing CNN features, auxiliary logits, 
                main logits, and updated sequence lengths.
        """
        cnn_feat, out_lengths = self.temporal_cnn(x_branch, lengths)

        B, C, T_prime = cnn_feat.shape
        device = cnn_feat.device

        time_steps = torch.arange(T_prime, device=device).unsqueeze(0)
        length_tensor = out_lengths.unsqueeze(1)

        mask = time_steps < length_tensor
        mask = mask.unsqueeze(1).expand_as(cnn_feat)

        cnn_feat = cnn_feat * mask

        aux_feat = cnn_feat.transpose(1, 2)
        aux_logits = self.gloss_head(aux_feat)

        lstm_out = self.context_lstm(cnn_feat, out_lengths)
        main_logits = self.gloss_head(lstm_out)

        return {
            "cnn_feat": cnn_feat,
            "aux_logits": aux_logits,
            "main_logits": main_logits,
            "logit_lengths": out_lengths,
        }

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, keep_prob: float = 1.0) -> dict:
        """
        Performs the complete forward pass, splitting features into main and inverse branches.

        Args:
            x (torch.Tensor): Input skeleton tensor of shape [Batch, Channels, Time, Vertices].
            lengths (torch.Tensor): 1D tensor of sequence lengths.
            keep_prob (float): Feature retention probability for the masking module. Defaults to 1.0.

        Returns:
            dict[str, dict]: A dictionary containing the processing results for both 'phi' and 'phi_inv' branches.
        """
        branches = self.STGCN(x, keep_prob=keep_prob)
        branch_phi = branches[:, 0, ...]
        branch_phi_inv = branches[:, 1, ...]

        out_phi = self._forward_branch(branch_phi, lengths)
        out_phi_inv = self._forward_branch(branch_phi_inv, lengths)

        return {
            "phi": out_phi,
            "phi_inv": out_phi_inv,
        }


class AttentivePooling(nn.Module):
    """
    Collapses a sequence of frames into a single feature vector using learned attention weights.
    Primarily used for Isolated Sign Language Recognition (ISLR).
    """
    
    def __init__(self, feat_dim: int):
        """
        Initializes the Attentive Pooling layer.

        Args:
            feat_dim (int): Dimensionality of the input features.
        """
        super().__init__()
        self.score = nn.Linear(feat_dim, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Applies attention-based pooling over the temporal dimension.

        Args:
            x (torch.Tensor): Input feature tensor of shape [Batch, Time, Channels].
            lengths (torch.Tensor): 1D tensor of valid sequence lengths.

        Returns:
            torch.Tensor: A temporally pooled 2D tensor of shape [Batch, Channels].
        """
        scores = self.score(x).squeeze(-1)

        B, T = scores.shape
        mask = torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        scores[~mask] = float("-inf")

        weights = F.softmax(scores, dim=-1).unsqueeze(-1)
        pooled = (x * weights).sum(dim=1)
        return pooled


class GlossClassifier(nn.Module):
    """
    Isolated gloss classifier model combining ST-GCN, attention pooling, and a linear head.
    """

    def __init__(self, num_classes: int, dropout: float = 0.2):
        """
        Initializes the ISLR Gloss Classifier.

        Args:
            num_classes (int): Total number of output gloss classes.
            dropout (float): Dropout probability. Defaults to 0.2.
        """
        super().__init__()

        self.STGCN = STGCNCoSign1s()
        self.pool = AttentivePooling(feat_dim=1024)
        self.head = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass to classify a single isolated sign.

        Args:
            x (torch.Tensor): Input skeleton tensor of shape [Batch, Channels, Time, Vertices].
            lengths (torch.Tensor): 1D tensor of valid sequence lengths.

        Returns:
            torch.Tensor: The output class logits of shape [Batch, NumClasses].
        """
        branches = self.STGCN(x, keep_prob=1.0)
        feat = branches[:, 0, :, :]
        feat = feat.transpose(1, 2)

        pooled = self.pool(feat, lengths)
        logits = self.head(pooled)
        return logits