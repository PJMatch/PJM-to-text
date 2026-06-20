"""
Module holding constant values and configuration for the PJMatch application.
"""

from pathlib import Path

TASKS_DIR: Path = Path("../mediapipe_tasks")

SLIDING_WINDOW_LENGTH_CSLR: int = 220
STRIDE_CSLR: int = 15

SLIDING_WINDOW_LENGTH_ISLR: int = 30
STRIDE_ISLR: int = 5
ISLR_CONFIDENCE_THRESHOLD: float = 0.75
ISLR_CUMULATIVE_THRESHOLD: float = 1.9

VOTE_THRESHOLD: int = 3

POSE_LEN: int = 33
FACE_LEN: int = 478
LH_LEN: int = 21
RH_LEN: int = 21
TOTAL_V: int = POSE_LEN + FACE_LEN + LH_LEN + RH_LEN

UI_FILE: str = "res/ui/main_window.ui"
TESTING_VIDEO_PATH: str = r"local"