## Visualization Tool (How to Use)

You can easily generate a GIF animation from any extracted MediaPipe `.npy` file using this visualization script.

### Basic Usage

Run the following command in your terminal. Make sure to replace the paths with the actual location of your `.npy` file:

```bash
python process_visualization_scripts/skeleton_visualization.py \
  --input process_visualization_scripts/9_97_20260324_194957.npy \
  --output process_visualization_scripts/samples/9_97_20260324_194957.gif