"""
Module containing the UI component for displaying predicted text.
"""

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QTextEdit


class OutputBox(QTextEdit):
    """
    Custom QTextEdit widget that serves as the sentence placeholder 
    for displaying neural network predictions in the GUI.
    """

    def __init__(self, parent=None) -> None:
        """
        Initializes the OutputBox with default styling and placeholder text.

        Args:
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent=parent)
        self.setStyleSheet("font-size: 32px; font-weight: bold;")
        self.setText("Waiting for predictions...")