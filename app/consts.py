"""Module holding const values."""

from pathlib import Path

TASKS_DIR = Path("../mediapipe_tasks")
SLIDING_WINDOW_LENGTH = 220  # in frames
STRIDE = 15
DOWNSAMPLING_FACTOR = 4  # CNN does 2x MaxPool1d → 4x temporal reduction
VOTE_THRESHOLD = 2  # gloss must appear in ≥ N overlapping windows

POSE_LEN = 33
FACE_LEN = 478
LH_LEN = 21
RH_LEN = 21
TOTAL_V = POSE_LEN + FACE_LEN + LH_LEN + RH_LEN

UI_FILE = "res/ui/main_window.ui"
TESTING_VIDEO_PATH = r"local"
