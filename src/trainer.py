"""Trainer module."""

import torch

from models.cosign_1s.cosign1s_stgcn import STGCNCoSign1s  # Import Twojej klasy


def test_run():
    """Test run."""
    batch_size = 2
    channels = 3
    timesteps = 50
    total_vertices = 553

    print("STGCNCoSign1s initialization")
    model = STGCNCoSign1s()
    model.eval()

    dummy_input = torch.randn(batch_size, channels, timesteps, total_vertices)
    print(f"input (raw npy): {dummy_input.shape}")

    with torch.no_grad():
        try:
            model(dummy_input)
            print("-" * 30)
            print("model processed data")

        except Exception as e:
            print(f"Error during forward pass: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    test_run()
