import torch

from stgcn.stgcn import STGCNCoSign1s


def test_edge_importance_gets_gradient():
    """Checks if edge_importance parameter receives gradients during backward pass."""
    # set the config_path to None to use the current config
    stgcn = STGCNCoSign1s(config_path=None)
    stgcn.train()

    dummy_input = torch.randn(2, 3, 60, 553)
    output = stgcn(dummy_input)

    loss = output.sum()
    loss.backward()

    found_edge_imp = False
    for name, param in stgcn.named_parameters():
        if "edge_importance" in name:
            found_edge_imp = True
            assert param.grad is not None, f"Gradient for {name} is None!"
            assert not torch.all(param.grad == 0), f"Gradient for {name} is all zeros!"

    assert found_edge_imp, "Could not find 'edge_importance' parameters in the model!"
