"""Module for frames extraction from a video."""

import cv2


def extract_frames(path: str, show_frames=False):
    """Extract frames from a video and saves them in a .jpg form."""
    video = cv2.VideoCapture(path)

    count = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        filename = f"frame_{count:04d}.jpg"

        cv2.imwrite("test_9_16_FHD/" + filename, frame)
        count += 1


if __name__ == "__main__":
    extract_frames("test_9_16_FHD.mp4")
