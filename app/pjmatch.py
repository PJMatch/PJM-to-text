"""Main PJMatch user desktop application."""

import queue
import sys

import consts
from camera_label import CameraLabel
from output_box import OutputBox
from PySide6.QtCore import QFile
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
)
from workers import AIWorker, VisionWorker, BielikWorker


class ModeSelectionDialog(QDialog):
    """A simple popup to choose the model before the app launches."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PJMatch - Select Mode")
        self.selected_mode = "CSLR"
        self.resize(300, 150)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Which model do you want to run?"))

        btn_cslr = QPushButton("CSLR (Continuous Sequence)")
        btn_cslr.clicked.connect(lambda: self.select_mode("CSLR"))
        layout.addWidget(btn_cslr)

        btn_islr = QPushButton("ISLR (Isolated Sliding Window)")
        btn_islr.clicked.connect(lambda: self.select_mode("ISLR"))
        layout.addWidget(btn_islr)

        self.setLayout(layout)

    def select_mode(self, mode):
        self.selected_mode = mode
        self.accept()


class PJMatchWindow(QMainWindow):
    """PJMatch app main window."""

    def __init__(self, mode="CSLR"):
        """Init function for PJMatchWindow."""
        super().__init__()
        self.mode = mode

        loader = QUiLoader()
        loader.registerCustomWidget(CameraLabel)
        loader.registerCustomWidget(OutputBox)

        ui_file = QFile(consts.UI_FILE)
        if not ui_file.open(QFile.ReadOnly):
            print(f"Cannot open {ui_file}: {ui_file.errorString()}")

        self.ui = loader.load(ui_file, self)
        ui_file.close()

        self.ai_queue = queue.Queue(maxsize=5)

        self.vision_worker = VisionWorker(
            shared_queue=self.ai_queue,
            mode=self.mode,
        )
        self.vision_worker.frame_ready.connect(self.ui.cameraLabel.update_frame)
        self.vision_worker.start()

        self.live_glosses = []
        self.sentence_history = []

        self.ai_worker = AIWorker(shared_queue=self.ai_queue, mode=self.mode)
        self.bielik_worker = BielikWorker()

        self.ai_worker.prediction_ready.connect(self.bielik_worker.add_task)
        self.ai_worker.prediction_ready.connect(self.update_live_glosses)
        
        self.bielik_worker.word_received.connect(self.update_sentence_history)

        self.ai_worker.start()
        self.bielik_worker.start()

        self.setCentralWidget(self.ui.centralwidget)
        self.resize(1000, 600)
        self.setWindowTitle(f"PJMatch - {self.mode} Mode")

    def update_live_glosses(self, new_gloss: str):
        new_gloss = new_gloss.strip()
        if new_gloss and new_gloss.lower() != "blank":
            self.live_glosses.append(new_gloss)
            self.refresh_ui()

    def update_sentence_history(self, new_sentence: str):
        self.sentence_history.append(new_sentence)
        self.live_glosses.clear()
        self.refresh_ui()

    def refresh_ui(self):
        gloss_str = " ".join(self.live_glosses)
        
        hist_list = []
        for i, sentence in enumerate(self.sentence_history, 1):
            hist_list.append(f"Zdanie {i}: {sentence}")
        hist_str = "\n".join(hist_list)
        
        final_text = f"BIEŻĄCE ZNAKI:\n{gloss_str}\n\nHISTORIA:\n{hist_str}"
        self.ui.sentenceHolder.setText(final_text)

    def closeEvent(self, event):
        """Stops threads on close."""
        self.vision_worker.stop()
        self.ai_worker.stop()
        self.bielik_worker.stop()
        event.accept()


if __name__ == "__main__":
    with open("prediction_log.txt", "w", encoding="utf-8") as f:
        f.write("")

    app = QApplication(sys.argv)

    dialog = ModeSelectionDialog()
    if dialog.exec() == QDialog.Accepted:
        window = PJMatchWindow(mode=dialog.selected_mode)
        window.show()
        sys.exit(app.exec())
    else:
        sys.exit(0)
