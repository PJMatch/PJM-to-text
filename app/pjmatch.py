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
    QMainWindow,
)
from workers import AIWorker, VisionWorker


class PJMatchWindow(QMainWindow):
    """PJMatch app main window."""

    def __init__(self):
        """Init function for PJMatchWindow."""
        super().__init__()
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
            testing_vid_path=consts.TESTING_VIDEO_PATH,
            window_width=consts.SLIDING_WINDOW_LENGTH,
        )
        self.vision_worker.frame_ready.connect(self.ui.cameraLabel.update_frame)
        self.vision_worker.start()

        self.ai_worker = AIWorker(shared_queue=self.ai_queue)
        self.ai_worker.prediction_ready.connect(self.ui.sentenceHolder.setText)
        self.ai_worker.start()

        self.setCentralWidget(self.ui.centralwidget)
        self.resize(1000, 600)
        self.setWindowTitle("PJMatch")

    def closeEvent(self, event):
        """Stops threads on close."""
        self.vision_worker.stop()
        self.ai_worker.stop()
        event.accept()


if __name__ == "__main__":
    with open("prediction_log.txt", "w", encoding="utf-8") as f:
        f.write("")

    app = QApplication(sys.argv)
    window = PJMatchWindow()
    window.show()
    sys.exit(app.exec())
