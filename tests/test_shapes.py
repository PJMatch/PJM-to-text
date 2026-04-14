"""UNIT TESTS."""

import torch

from stgcn.stgcn import STGCNCoSign1s


def test_stgcn_forward_shape():
    """Unit test for ST-GCN input:output shapes and checks if no values are NaN."""
    # set the config_path to None to use the current config
    stgcn = STGCNCoSign1s(config_path=None)

    curr_config = stgcn.gso_generator.config

    B = 4
    C = 3
    T = 60
    V = 0
    for _, vertecies in curr_config.items():
        V += len(vertecies)

    V = 553
    dummy_input = torch.randn(B, C, T, V)
    output = stgcn(dummy_input)

    expected_shape = (B, 2, 1024, T)

    assert output.shape == expected_shape, (
        f"Expected output shape {expected_shape}, but got {output.shape}"
    )

    assert not torch.isnan(output).any()
