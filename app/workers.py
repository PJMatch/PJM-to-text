"""Workers module."""

import queue
import time

import consts
import cv2
from mp_node import MPNode
from pjm_nn_node import GlossTracker, PJMPredictor, SentenceSmoother
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
            self.camera = cv2.VideoCapture(0)

        self.fps = 30

        self.frame_delay_ms = int(1000 / self.fps)

        self.frames_since_last_predict = 0
        self.stride = 15

        self.frame_count = 0
        self.playback_start_time = None

        self.absolute_frame = 0

    def run(self):
        """Runs VisionWorker.

        Reads frames, sends them to be displayed, has the MediaPipe node do the inference every
        stride.
        """
        self.playback_start_time = time.time()
        while self.running:
            ret, frame = self.camera.read()
            if not ret:
                continue

            self.frame_ready.emit(frame)
            self.absolute_frame += 1

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

            window_start = self.absolute_frame - len(window_chunk)

            if not self.shared_queue.full():
                self.shared_queue.put((window_chunk, window_start))

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
    Pipeline: predict → GlossTracker → SentenceSmoother → UI
    """

    prediction_ready = Signal(str)

    def __init__(self, shared_queue):
        super().__init__()
        self.running = True
        self.shared_queue = shared_queue
        self.predictor = PJMPredictor()

        self.tracker = GlossTracker()
        self.smoother = SentenceSmoother()

    def run(self):
        while self.running:
            try:
                window_chunk, window_start = self.shared_queue.get(timeout=1)
            except queue.Empty:
                continue

            gloss_predictions = self.predictor.predict(window_chunk, window_start)

            voted_string = self.tracker.vote(gloss_predictions)

            if voted_string:
                print(f"DEBUG Tracker: {voted_string}")

                final_sentence = self.smoother.process(voted_string)

                if final_sentence:
                    print(f"\n--- EMITTING TO UI: {final_sentence} ---\n")
                    self.prediction_ready.emit(final_sentence)

    def stop(self):
        leftover = self.smoother._commit()
        if leftover:
            self.prediction_ready.emit(leftover)

        self.running = False
        self.quit()
        self.wait()
