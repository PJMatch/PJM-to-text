"""Workers module."""

import queue
import time

import consts
import cv2
from mp_node import MPNode
from pjm_nn_node import PJMPredictor, SentenceSmoother
from PySide6.QtCore import QThread, Signal


class VisionWorker(QThread):
    """Vision worker.

    Manages frames and MediaPipe extraction.
    """

    frame_ready = Signal(object)

    def __init__(self, shared_queue, window_width, testing_vid_path=None):
        """Constructor of the VisionWorker."""
        super().__init__()
        self.running = True
        self.mp_node = MPNode()
        self.shared_queue = shared_queue

        self.target_window_width = window_width

        self.frames_since_last_predict = 0
        self.stride = consts.STRIDE

        self.video_path = testing_vid_path
        if self.video_path is not None:
            self.camera = cv2.VideoCapture(self.video_path)
        else:
            print("huj")
            self.camera = cv2.VideoCapture(0)

        self.fps = 30

        self.frame_delay_ms = int(1000 / self.fps)

        self.frames_since_last_predict = 0
        self.stride = 15

        self.frame_count = 0
        self.playback_start_time = None

    def run(self):
        """Runs VisionWorker QThread.

        Reads frames, sends them to be displayed, has the MediaPipe node do the inference every
        stride,
        """
        self.playback_start_time = time.time()
        while self.running:
            ret, frame = self.camera.read()
            if not ret:
                continue

            self.frame_ready.emit(frame)

            # TODO: Prepare frame for MediaPipe???
            self.mp_node.receive_frame(frame)

            if len(self.mp_node.sliding_window.frames) != self.target_window_width:
                # sleep to not check every milisecond and take up processor
                self.msleep(10)
                continue

            self.frames_since_last_predict += 1
            if self.frames_since_last_predict < self.stride:
                continue

            self.frames_since_last_predict = 0
            window_chunk = self.mp_node.sliding_window.get_window()

            if not self.shared_queue.full():
                self.shared_queue.put(window_chunk)

            if self.video_path:
                target_time = self.playback_start_time + (self.frame_count / self.fps)
                current_time = time.time()

                sleep_time_seconds = target_time - current_time

                if sleep_time_seconds > 0:
                    self.msleep(int(sleep_time_seconds * 1000))

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

        self.smoother = SentenceSmoother()

    def run(self):
        """Runs the AIWorker QThread."""
        while self.running:
            try:
                window_chunk = self.shared_queue.get(timeout=1)
                raw_text = self.predictor.predict(window_chunk)

                clean_sentence = self.smoother.process(raw_text)

                if clean_sentence:
                    print(clean_sentence)
                    self.prediction_ready.emit(clean_sentence)
            except queue.Empty:
                # no data in the queue after one second
                continue

    def stop(self):
        """Stops the thread."""
        self.running = False
        self.quit()
        self.wait()
