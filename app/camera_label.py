"""
Module containing the UI component for displaying the camera feed.
"""

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel


class CameraLabel(QLabel):
    """
    Custom QLabel widget that handles the display and scaling of raw 
    webcam frames extracted by the computer vision backend.
    """

    def __init__(self, parent=None) -> None:
        """
        Initializes the CameraLabel with default styling and placeholder text.

        Args:
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent=parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #111; color: #fff; font-size: 18px;")
        self.setText("Initializing camera...")

    def update_frame(self, frame: np.ndarray) -> None:
        """
        Callback function connected to the VisionWorker's signal.
        Converts a raw OpenCV BGR frame to a Qt-compatible QPixmap and displays it.

        Args:
            frame (np.ndarray): The raw image frame array from OpenCV.

        Returns:
            None
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.width(),
            self.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.setPixmap(scaled_pixmap)