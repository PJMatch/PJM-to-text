"""Workers module."""

import queue

import cv2
from mp_node import MPNode
from pjm_nn_node import PJMPredictor
from PySide6.QtCore import QThread, Signal


class VisionWorker(QThread):
    """Vision worker.

    Manages frames and MediaPipe extraction.
    """

    frame_ready = Signal(object)

    def __init__(self, shared_queue):
        """Constructor of the VisionWorker."""
        super().__init__()
        self.running = True
        self.mp_node = MPNode()
        self.shared_queue = shared_queue

        self.camera = cv2.VideoCapture(0)

        self.frames_since_last_predict = 0
        self.stride = 15

    def run(self):
        """Runs VisionWorker QThread.

        Reads frames, sends them to be displayed, has the MediaPipe node do the inference every
        stride,
        """
        while self.running:
            ret, frame = self.camera.read()
            if not ret:
                continue

            self.frame_ready.emit(frame)

            # TODO: Prepare frame for MediaPipe???
            self.mp_node.receive_frame(frame)
            if len(self.mp_node.sliding_window.frames) != 120:
                continue

            self.frames_since_last_predict += 1
            if self.frames_since_last_predict < self.stride:
                continue

            self.frames_since_last_predict = 0
            window_chunk = self.mp_node.sliding_window.get_window()

            if not self.shared_queue.full():
                self.shared_queue.put(window_chunk)

    def stop(self):
        """Stops the thread."""
        self.running = False
        self.camera.release()
        self.quit()
        self.wait()


class AIWorker(QThread):
    """AI worker.

    Manages the PJM predictor and sends the output to display.
    """

    prediction_ready = Signal(str)

    def __init__(self, shared_queue):
        """Constructor of the AIWorker."""
        super().__init__()
        self.running = True
        self.shared_queue = shared_queue
        self.predictor = PJMPredictor()

    def run(self):
        """Runs the AIWorker QThread."""
        while self.running:
            try:
                window_chunk = self.shared_queue.get(timeout=1)
                predicted_gloss = self.predictor.predict(window_chunk)

                if predicted_gloss and predicted_gloss != "<blank>":
                    self.prediction_ready.emit(predicted_gloss)
            except queue.Empty:
                # no data in the queue after one second
                continue

    def stop(self):
        """Stops the thread."""
        self.running = False
        self.quit()
        self.wait()
