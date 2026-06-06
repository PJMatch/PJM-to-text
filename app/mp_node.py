"""Module containing MediaPipe inference node logic."""

import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import consts
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def _safe_part(raw, expected_len):
    arr = np.array(raw, dtype=np.float32) if len(raw) > 0 else np.zeros((0, 4), dtype=np.float32)
    arr = arr.reshape(-1, 4)
    if arr.shape[0] == 0:
        return np.zeros((expected_len, 4), dtype=np.float32)
    if arr.shape[0] < expected_len:
        arr = np.pad(arr, ((0, expected_len - arr.shape[0]), (0, 0)))
    return arr[:expected_len]


def format_frame_for_nn(frame_dict):
    """Formats extracted raw keypoints for the nn."""
    pose = _safe_part(frame_dict.get("pose", []), consts.POSE_LEN)
    face = _safe_part(frame_dict.get("face", []), consts.FACE_LEN)
    lh = _safe_part(frame_dict.get("lh", []), consts.LH_LEN)
    rh = _safe_part(frame_dict.get("rh", []), consts.RH_LEN)

    combined = np.concatenate([pose, face, lh, rh], axis=0)
    # Output x, y, confidence (channels 0, 1, 3) instead of x, y, z (channels 0, 1, 2)
    return combined[:, [0, 1, 3]]


@dataclass
class SlidingWindow:
    """Dataclass holding the current window for MediaPipe inference."""

    max_len: int = consts.SLIDING_WINDOW_LENGTH
    frames: deque = field(default_factory=lambda: deque(maxlen=consts.SLIDING_WINDOW_LENGTH))

    def append(self, frame) -> None:
        """Appends the frames deque if not full, else appends and drops oldest frame."""
        self.frames.append(frame)

    def get_window(self):
        """Returns the full window."""
        return list(self.frames)


class MPNode:
    """MediaPipe inference node class."""

    def __init__(self):
        """Constructor of MPNode class."""
        face_model_path = consts.TASKS_DIR / "face_landmarker_v2_with_blendshapes.task"
        face_base_options = python.BaseOptions(model_asset_path=str(face_model_path))
        face_options = vision.FaceLandmarkerOptions(
            base_options=face_base_options,
            running_mode=vision.RunningMode.VIDEO,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        self.face_detector = vision.FaceLandmarker.create_from_options(face_options)

        pose_model_path = consts.TASKS_DIR / "pose_landmarker_lite.task"
        pose_base_options = python.BaseOptions(model_asset_path=str(pose_model_path))
        pose_options = vision.PoseLandmarkerOptions(
            base_options=pose_base_options,
            running_mode=vision.RunningMode.VIDEO,
            output_segmentation_masks=False,
        )
        self.pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

        self.extractor_thread = ThreadPoolExecutor(max_workers=3)

        hand_model_path = consts.TASKS_DIR / "hand_landmarker.task"
        hand_base_options = python.BaseOptions(model_asset_path=str(hand_model_path))
        hand_options = vision.HandLandmarkerOptions(
            base_options=hand_base_options, running_mode=vision.RunningMode.VIDEO, num_hands=2
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(hand_options)

        self.sliding_window = SlidingWindow()

    def __del__(self):
        """Destructor of MPNode class."""
        self.face_detector.close()
        self.pose_detector.close()
        self.hand_detector.close()
        self.extractor_thread.shutdown(wait=False)

    def extract_raw_keypoints(self, pose_result, hand_result, face_result):
        """Extracts visible landmarks into a structured dictionary without zero-padding.

        Confidence score is extracted from pose visibility.
        Face and hands default to 1.0 if detected.
        Missing components remain as empty lists.
        """
        frame_data = {"pose": [], "face": [], "lh": [], "rh": []}

        # Pose (33 points)
        if pose_result and pose_result.pose_landmarks:
            frame_data["pose"] = [
                [
                    lm.x,
                    lm.y,
                    lm.z,
                    getattr(lm, "visibility", 1.0),
                ]  # meidapipe native visibility score
                for lm in pose_result.pose_landmarks[0]
            ]

        # Face (478 points)
        if face_result and face_result.face_landmarks:
            frame_data["face"] = [
                [lm.x, lm.y, lm.z, 1.0]  # confidence score set to 1.0 by default
                for lm in face_result.face_landmarks[0]
            ][:478]  # cutoff for limit

        # Hands (21 points each)
        if hand_result and hand_result.hand_landmarks:
            for idx, hand_landmarks in enumerate(hand_result.hand_landmarks):
                handedness = hand_result.handedness[idx][0].category_name
                coords = [
                    [lm.x, lm.y, lm.z, 1.0]  # confidence score set to 1.0 by default
                    for lm in hand_landmarks
                ]
                if handedness == "Left":
                    frame_data["lh"] = coords
                elif handedness == "Right":
                    frame_data["rh"] = coords

        return frame_data

    def run_mp_inference(self, frame):
        """Runs MediaPipe inference on a singular frame.

        Returns:
            dict: {"face_result", "pose_result", "hand_result"}
        """
        last_timestamp_ms = int(time.time() * 1000)

        # small_frame = cv2.resize(frame, (640, 480))
        # downsized frame gives approx. 1-2 FPS improvement so for now useless

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        timestamp_ms = int(time.time() * 1000)
        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1
        last_timestamp_ms = timestamp_ms

        future_face = self.extractor_thread.submit(
            self.face_detector.detect_for_video, mp_image, timestamp_ms
        )
        future_pose = self.extractor_thread.submit(
            self.pose_detector.detect_for_video, mp_image, timestamp_ms
        )
        future_hands = self.extractor_thread.submit(
            self.hand_detector.detect_for_video, mp_image, timestamp_ms
        )

        face_result = future_face.result()
        pose_result = future_pose.result()
        hand_result = future_hands.result()

        return {
            "face_result": face_result,
            "pose_result": pose_result,
            "hand_result": hand_result,
        }

    def get_keypoints_from_frame(self, frame):
        """Returns a numpy array with all keypoints."""
        inf_res = self.run_mp_inference(frame)
        raw_keypoints = self.extract_raw_keypoints(
            inf_res["pose_result"], inf_res["hand_result"], inf_res["face_result"]
        )
        return raw_keypoints

    def receive_frame(self, frame):
        """Receives a singular frame, runs inference and adds it to the sliding window."""
        raw_keypoints = self.get_keypoints_from_frame(frame)
        nn_ready_frame = format_frame_for_nn(raw_keypoints)
        self.sliding_window.append(nn_ready_frame)


if __name__ == "__main__":
    mp_node = MPNode()
