"""Module containing MediaPipe inference node logic."""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils

TASKS_DIR = Path("../mediapipe_tasks")


# TODO: get these functions from the mediapipe_test module


latest_face_result = None
latest_pose_result = None
latest_hand_result = None


class MPNode:
    """MediaPipe inference node class."""

    def __init__(self):
        face_model_path = TASKS_DIR / "face_landmarker_v2_with_blendshapes.task"
        face_base_options = python.BaseOptions(model_asset_path=str(face_model_path))
        face_options = vision.FaceLandmarkerOptions(
            base_options=face_base_options,
            running_mode=vision.RunningMode.VIDEO,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        self.face_detector = vision.FaceLandmarker.create_from_options(face_options)

        pose_model_path = TASKS_DIR / "pose_landmarker_lite.task"
        pose_base_options = python.BaseOptions(model_asset_path=str(pose_model_path))
        pose_options = vision.PoseLandmarkerOptions(
            base_options=pose_base_options,
            running_mode=vision.RunningMode.VIDEO,
            output_segmentation_masks=False,
        )
        self.pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

        hand_model_path = TASKS_DIR / "hand_landmarker.task"
        hand_base_options = python.BaseOptions(model_asset_path=str(hand_model_path))
        hand_options = vision.HandLandmarkerOptions(
            base_options=hand_base_options, running_mode=vision.RunningMode.VIDEO, num_hands=2
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(hand_options)

    def __del__(self):
        """Destructor of MPNode class."""
        self.face_detector.close()
        self.pose_detector.close()
        self.hand_detector.close()

    def extract_raw_keypoints(pose_result, hand_result, face_result):
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
        feature_extractor_thread = ThreadPoolExecutor(max_workers=3)

        last_timestamp_ms = int(time.time() * 1000)

        # small_frame = cv2.resize(frame, (640, 480))
        # downsized frame gives approx. 1-2 FPS improvement so for now useless

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        timestamp_ms = int(time.time() * 1000)
        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1
        last_timestamp_ms = timestamp_ms

        future_face = feature_extractor_thread.submit(
            self.face_detector.detect_for_video, mp_image, timestamp_ms
        )
        future_pose = feature_extractor_thread.submit(
            self.pose_detector.detect_for_video, mp_image, timestamp_ms
        )
        future_hands = feature_extractor_thread.submit(
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
        """Returns a numpy array with all keypoints - ready for PJM nn inference."""
        inf_res = self.run_mp_inference(frame)
        raw_keypoints = self.extract_raw_keypoints(
            inf_res["pose_result"], inf_res["hand_result"], inf_res["face_result"]
        )
        return raw_keypoints


if __name__ == "__main__":
    mp_node = MPNode()
    mp_node.synchronous_detect()
