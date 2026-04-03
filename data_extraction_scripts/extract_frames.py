"""Module for frames extraction from a video."""

import cv2


def extract_frames(path: str, show_frames=False) -> list:
    """Extract frames from a video."""
    video = cv2.VideoCapture(path)

    frames = []
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        frames.append(frame)

    return frames


if __name__ == "__main__":
    extract_frames("test_9_16_FHD.mp4")
