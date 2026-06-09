"""Module for the OutputBox logic."""

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QTextEdit


class OutputBox(QTextEdit):
    """Class for the sentence placeholder."""

    def __init__(self, parent=None):
        """Init function for OutputBox."""
        super().__init__()
        self.setStyleSheet("font-size: 32px; font-weight: bold;")

        self.setText("Waiting for predictions...")
