"""UNIT TESTS."""

import torch


def test_causal_conv_no_future_peeking():
    """Ensures that changing a future frame does not alter past outputs in CausalConv."""
    from stgcn.stgcn_src.layers import CausalConv1d

    conv = CausalConv1d(in_channels=64, out_channels=64, kernel_size=3, enable_padding=True)
    conv.eval()

    seq_A = torch.randn(1, 64, 100)

    seq_B = seq_A.clone()
    seq_B[:, :, -1] += 100.0

    out_A = conv(seq_A)
    out_B = conv(seq_B)

    assert torch.allclose(out_A[:, :, 0], out_B[:, :, 0], atol=1e-6), (
        "Causal padding failed: past was affected by future!"
    )


def test_temporal_cnn_downsampling():
    """Verifies that the temporal dimension is reduced by a factor of 4 (two poolings)."""
    from model import CoSignTemporalCNN

    cnn = CoSignTemporalCNN(in_dim=1024, hidden_dim=1024)

    B, C, T = 4, 1024, 240
    dummy_input = torch.randn(B, C, T)
    lengths = torch.tensor([240, 200, 100, 50])

    out, out_lengths = cnn(dummy_input, lengths)

    # tensor shape should be exactly T // 4 due to two stride=2 poolings
    expected_T = 240 // 4
    assert out.shape == (B, C, expected_T), f"Expected T={expected_T}, got {out.shape[2]}"

    expected_lengths = lengths // 4
    assert torch.equal(out_lengths, expected_lengths), (
        f"Lengths calculation mismatch: {out_lengths} vs {expected_lengths}"
    )


def test_gloss_head_cosine_bounds():
    """Ensures the cosine similarity is bounded between [-1, 1] before scaling."""
    from model import SharedGlossHead

    head = SharedGlossHead(feat_dim=1024, vocab_size=1086)
    dummy_features = torch.randn(8, 1024)

    logits = head(dummy_features)

    raw_similarity = logits / head.scale

    assert torch.all(raw_similarity >= -1.0001), "Cosine similarity below -1"
    assert torch.all(raw_similarity <= 1.0001), "Cosine similarity above 1"
