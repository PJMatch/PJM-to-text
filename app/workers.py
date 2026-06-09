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

    def __init__(self, shared_queue, mode="CSLR", testing_vid_path=None):
        """Constructor of the VisionWorker."""
        super().__init__()
        self.running = True
        self.shared_queue = shared_queue
        self.mode = mode

        if self.mode == "ISLR":
            self.target_window_width = consts.SLIDING_WINDOW_LENGTH_ISLR
            self.stride = consts.STRIDE_ISLR
        else:
            self.target_window_width = consts.SLIDING_WINDOW_LENGTH_CSLR
            self.stride = consts.STRIDE_CSLR

        self.mp_node = MPNode(max_window_len=self.target_window_width)

        self.video_path = testing_vid_path
        if self.video_path is not None:
            self.camera = cv2.VideoCapture(self.video_path)
        else:
            self.camera = cv2.VideoCapture(0)

        self.fps = 30
        self.frame_delay_ms = int(1000 / self.fps)

        self.frames_since_last_predict = 0
        self.frame_count = 0
        self.absolute_frame = 0
        self.playback_start_time = None

    def run(self):
        """Runs VisionWorker.

        Reads frames, sends them to be displayed, enforces playback speed,
        and has the MediaPipe node do the inference every stride.
        """
        self.playback_start_time = time.time()

        while self.running:
            ret, frame = self.camera.read()
            if not ret:
                continue

            self.absolute_frame += 1
            self.frame_count += 1

            self.frame_ready.emit(frame)
            self.mp_node.receive_frame(frame)

            if self.video_path:
                target_time = self.playback_start_time + (self.frame_count / self.fps)
                current_time = time.time()
                sleep_time_seconds = target_time - current_time

                if sleep_time_seconds > 0:
                    self.msleep(int(sleep_time_seconds * 1000))

            if len(self.mp_node.sliding_window) != self.target_window_width:
                continue

            self.frames_since_last_predict += 1
            if self.frames_since_last_predict < self.stride:
                continue

            self.frames_since_last_predict = 0
            window_chunk = list(self.mp_node.sliding_window)
            window_start = self.absolute_frame - len(window_chunk)

            if not self.shared_queue.full():
                self.shared_queue.put((window_chunk, window_start))

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

    def __init__(self, shared_queue, mode="CSLR"):
        super().__init__()
        self.running = True
        self.shared_queue = shared_queue
        self.mode = mode

        self.predictor = PJMPredictor(mode=self.mode)
        self.tracker = GlossTracker(mode=self.mode)
        self.smoother = SentenceSmoother()

        self.last_islr_word = None
        self.candidate_word = None
        self.candidate_count = 0
        self.required_confirmations = 2

    def run(self):
        while self.running:
            try:
                window_chunk, window_start = self.shared_queue.get(timeout=1)
            except queue.Empty:
                continue

            gloss_predictions = self.predictor.predict(window_chunk)

            voted_string = self.tracker.vote(gloss_predictions)

            if self.mode == "CSLR":
                if voted_string:
                    print(f"DEBUG Tracker: {voted_string}")
                    final_sentence = self.smoother.process(voted_string)

                    if final_sentence:
                        print(f"\n--- EMITTING TO UI: {final_sentence} ---\n")
                        self.prediction_ready.emit(final_sentence)
                        with open("prediction_log.txt", "a", encoding="utf-8") as f:
                            f.write(final_sentence + "\n")

            else:
                if voted_string:
                    print(f"DEBUG Tracker: {voted_string}")

                    if voted_string != self.last_islr_word:
                        if voted_string == self.candidate_word:
                            self.candidate_count += 1
                        else:
                            self.candidate_word = voted_string
                            self.candidate_count = 1

                        if self.candidate_count >= self.required_confirmations:
                            self.last_islr_word = voted_string
                            self.candidate_word = None
                            self.candidate_count = 0

                            if voted_string == "blank":
                                voted_string = " "
                            print(f"\n--- EMITTING TO UI: {voted_string} ---\n")
                            self.prediction_ready.emit(voted_string)
                            with open("prediction_log.txt", "a", encoding="utf-8") as f:
                                f.write(voted_string + "\n")
                else:
                    self.last_islr_word = None
                    self.candidate_word = None
                    self.candidate_count = 0

                # if voted_string:
                #     print(f"DEBUG Tracker: {voted_string}")

                #     self.prediction_ready.emit(voted_string)
                #     with open("prediction_log.txt", "a", encoding="utf-8") as f:
                #         f.write(voted_string + "\n")

    def stop(self):
        if self.mode == "CSLR":
            leftover = self.smoother._commit()
            if leftover:
                self.prediction_ready.emit(leftover)

        self.running = False
        self.quit()
        self.wait()
