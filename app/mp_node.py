"""
Module containing the Google MediaPipe computer vision integration.

Handles the asynchronous extraction of body, hand, and face spatial coordinates
from raw image frames to construct a unified skeleton representation.
"""

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


def _safe_part(raw: list, expected_len: int) -> np.ndarray:
    """
    Ensures that an extracted body part array strictly matches the expected vertex count by padding or truncating.

    Args:
        raw (list): The raw extracted landmark points.
        expected_len (int): The target length (e.g., 21 for hands).

    Returns:
        np.ndarray: A padded or truncated 2D numpy array of shape (expected_len, 4).
    """
    arr = np.array(raw, dtype=np.float32) if len(raw) > 0 else np.zeros((0, 4), dtype=np.float32)
    arr = arr.reshape(-1, 4)
    if arr.shape[0] == 0:
        return np.zeros((expected_len, 4), dtype=np.float32)
    if arr.shape[0] < expected_len:
        arr = np.pad(arr, ((0, expected_len - arr.shape[0]), (0, 0)))
    return arr[:expected_len]


def format_frame_for_nn(frame_dict: dict) -> np.ndarray:
    """
    Formats the raw MediaPipe dictionaries into a continuous sequence tensor for the neural network.

    Args:
        frame_dict (dict): Dictionary holding raw extracted nodes ('pose', 'face', 'lh', 'rh').

    Returns:
        np.ndarray: A concatenated array of shape (TOTAL_V, 3) containing [X, Y, Visibility].
    """
    pose = _safe_part(frame_dict.get("pose", []), consts.POSE_LEN)
    face = _safe_part(frame_dict.get("face", []), consts.FACE_LEN)
    lh = _safe_part(frame_dict.get("lh", []), consts.LH_LEN)
    rh = _safe_part(frame_dict.get("rh", []), consts.RH_LEN)

    combined = np.concatenate([pose, face, lh, rh], axis=0)
    return combined[:, [0, 1, 3]]


class MPNode:
    """
    MediaPipe processing class. Uses thread pools to run heavy face, pose, 
    and hand detection models concurrently.
    """

    def __init__(self, max_window_len: int) -> None:
        """
        Initializes the MediaPipe detectors and the thread execution pool.

        Args:
            max_window_len (int): Maximum length of the sliding window queue holding processed frames.
        """
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

        self.sliding_window = deque(maxlen=max_window_len)

    def __del__(self) -> None:
        """Destructor to securely release detector resources and shutdown the thread pool."""
        if hasattr(self, "face_detector"):
            self.face_detector.close()
        if hasattr(self, "pose_detector"):
            self.pose_detector.close()
        if hasattr(self, "hand_detector"):
            self.hand_detector.close()
        if hasattr(self, "extractor_thread"):
            self.extractor_thread.shutdown(wait=False)

    def extract_raw_keypoints(self, pose_result, hand_result, face_result) -> dict:
        """
        Translates raw MediaPipe objects into a structured dictionary of coordinates.

        Args:
            pose_result: Output from PoseLandmarker.
            hand_result: Output from HandLandmarker.
            face_result: Output from FaceLandmarker.

        Returns:
            dict: Processed coordinates grouped by body part.
        """
        frame_data = {"pose": [], "face": [], "lh": [], "rh": []}

        if pose_result and pose_result.pose_landmarks:
            frame_data["pose"] = [
                [
                    lm.x,
                    lm.y,
                    lm.z,
                    getattr(lm, "visibility", 1.0),
                ]
                for lm in pose_result.pose_landmarks[0]
            ]

        if face_result and face_result.face_landmarks:
            frame_data["face"] = [[lm.x, lm.y, lm.z, 1.0] for lm in face_result.face_landmarks[0]][
                :478
            ]

        if hand_result and hand_result.hand_landmarks:
            for idx, hand_landmarks in enumerate(hand_result.hand_landmarks):
                handedness = hand_result.handedness[idx][0].category_name
                coords = [[lm.x, lm.y, lm.z, 1.0] for lm in hand_landmarks]
                if handedness == "Left":
                    frame_data["lh"] = coords
                elif handedness == "Right":
                    frame_data["rh"] = coords

        return frame_data

    def run_mp_inference(self, frame: np.ndarray) -> dict:
        """
        Submits a single image frame to the asynchronous thread pool for concurrent evaluation.

        Args:
            frame (np.ndarray): Raw BGR frame from the camera.

        Returns:
            dict: Raw result objects from face, pose, and hand tracking models.
        """
        last_timestamp_ms = int(time.time() * 1000)

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

    def get_keypoints_from_frame(self, frame: np.ndarray) -> dict:
        """
        Executes inference and standardizes output for a given frame.

        Args:
            frame (np.ndarray): The raw video frame.

        Returns:
            dict: The extracted keypoints.
        """
        inf_res = self.run_mp_inference(frame)
        raw_keypoints = self.extract_raw_keypoints(
            inf_res["pose_result"], inf_res["hand_result"], inf_res["face_result"]
        )
        return raw_keypoints

    def receive_frame(self, frame: np.ndarray) -> None:
        """
        Accepts a frame, processes its keypoints, formats them for PyTorch, 
        and queues them into the rolling sliding window buffer.

        Args:
            frame (np.ndarray): The raw video frame.
            
        Returns:
            None
        """
        raw_keypoints = self.get_keypoints_from_frame(frame)
        nn_ready_frame = format_frame_for_nn(raw_keypoints)
        self.sliding_window.append(nn_ready_frame)