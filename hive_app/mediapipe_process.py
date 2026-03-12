import urllib.request
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
from typing import Callable, Optional

POSE_MODEL = 'pose_landmarker_lite.task'
HAND_MODEL = 'hand_landmarker.task'
FACE_MODEL = 'face_landmarker_v2_with_blendshapes.task'

MODELS = {
    POSE_MODEL: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    HAND_MODEL: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    FACE_MODEL: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
}
def download_models():
    for filename, url in MODELS.items():
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filename)
            print(f"Done: {filename}")

def extract_keypoints(pose_result, hand_result, face_result):
    """
    Extracts and concatenates landmarks into a single feature vector of 1659 values.
    Every landmark was multiplied by 3 (coordinates (X,Y,Z))
    First 99 indices are for pose, next 1434 for face, next 63 for left hand and last 63 for right hand
    Implements robust zero-padding to keep consistency in array shapes.
    """
    # Pose = 99 indices
    pose_data = np.zeros(33 * 3)
    if pose_result and pose_result.pose_landmarks:
        pose_data = np.array([[lm.x, lm.y, lm.z] for lm in pose_result.pose_landmarks[0]]).flatten()

    # Face = 1434 indices
    face_data = np.zeros(478 * 3)
    if face_result and face_result.face_landmarks:
        coords = np.array([[lm.x, lm.y, lm.z] for lm in face_result.face_landmarks[0]]).flatten()
        if len(coords) <= 1434:
            face_data[:len(coords)] = coords
        else:
            face_data = coords[:1434]

    # Hands = 63 indices (times two hands) = 126
    lh_data = np.zeros(21 * 3)
    rh_data = np.zeros(21 * 3)
    if hand_result and hand_result.hand_landmarks:
        for idx, hand_landmarks in enumerate(hand_result.hand_landmarks):
            handedness = hand_result.handedness[idx][0].category_name
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks]).flatten()
            if handedness == 'Left':
                lh_data = coords
            elif handedness == 'Right':
                rh_data = coords

    # Concatenate of features: 99 + 1434 + 63 + 63 = 1659
    return np.concatenate([pose_data, face_data, lh_data, rh_data])


def process_video(path: str, progress_callback: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
    """
    Processes a single video file and returns extracted keypoints as a NumPy array.
    Shape: (N_frames, 1659)
    """
    download_models()

    pose_detector = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=POSE_MODEL),
            running_mode=vision.RunningMode.VIDEO)
    )
    hand_detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=HAND_MODEL),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2)
    )
    face_detector = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=FACE_MODEL),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1)
    )

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_idx = 0
    sequence_data = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Conversion BGR->RGB
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

            # Global timestamp must increase monotonically
            timestamp_ms = int(frame_idx * 1000 / fps)

            # Inference
            pose_result = pose_detector.detect_for_video(mp_image, timestamp_ms)
            hand_result = hand_detector.detect_for_video(mp_image, timestamp_ms)
            face_result = face_detector.detect_for_video(mp_image, timestamp_ms)

            # Feature extraction with zero-padding
            keypoints = extract_keypoints(pose_result, hand_result, face_result)
            sequence_data.append(keypoints)

            frame_idx += 1

            if progress_callback is not None:
                progress_callback(frame_idx, total_frames)
    finally:
        cap.release()
        pose_detector.close()
        hand_detector.close()
        face_detector.close()

    # Sequence saved as a binary NumPy array
    return np.array(sequence_data, dtype=np.float32)