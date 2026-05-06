# ST-GCN Debug Sandbox & Visualizer

An isolated testing environment for analyzing data flow and visualizing features in the `CoSign1S` (ST-GCN) model for the PJMatch.

This environment allows you to safely "check inside" the model, generate memory dumps (checkpoints) from various network layers, and visualize them in an animated GIF.

##  Directory Structure

For the environment to work correctly, your `debug` folder should look like this:
```text
process_visualization_scripts/debug/
├── stgcn/                      # COPIED production stgcn folder
├── 9_97_20260324_194957.npy    # Example file with raw data (MediaPipe dicts)
├── run_sandbox.py              # Main script to run the model
├── model_debug.py              # Isolated model class with hooked-up export
├── stgcn_debug.py              # Isolated ST-GCN (calculates and exports anchoring)
├── debug_exporter.py           # Tools for saving tensors (.npy)
├── visualize_debug_all.py      # Script generating the GIF file
└── README.md                   # This file