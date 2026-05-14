from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QTextEdit


class OutputBox(QTextEdit):
    def __init__(self, parent=None):
        super().__init__()

        self.log_counter = 0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_output)
        self.timer.start(1000)

    def update_output(self):
        self.log_counter += 1
        output_sentence = f"Log entry: {self.log_counter}. Logged successfuly"
        self.setText(output_sentence)
