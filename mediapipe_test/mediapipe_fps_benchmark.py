"""Module for synchronous Mediapipe benchmark."""

# most of the funcitions' code has been copied from the official mediapipe website and later only
# finetuned for our usecase, source: https://ai.google.dev/edge/mediapipe/solutions/guide

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils
from mediapipe_detection import (
    draw_face_landmarks_on_image,
    draw_hand_landmarks_on_image,
    draw_pose_landmarks_on_image,
)


def synchronous_detect_benchmark(show_frames=False):
    """Synchronously detects landmarks and draws them on device camera livestream.

    This function provides a synchronous version of detection that displays detected
    keypoints on the live camera feedback sequentially frame after frame without skiping frames.

    Returns:
        None
    """
    face_base_options = python.BaseOptions(
        model_asset_path="face_landmarker_v2_with_blendshapes.task"
    )
    face_options = vision.FaceLandmarkerOptions(
        base_options=face_base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    face_detector = vision.FaceLandmarker.create_from_options(face_options)

    pose_base_options = python.BaseOptions(model_asset_path="pose_landmarker_lite.task")
    pose_options = vision.PoseLandmarkerOptions(
        base_options=pose_base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_segmentation_masks=False,
    )
    pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

    hand_base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
    hand_options = vision.HandLandmarkerOptions(
        base_options=hand_base_options, running_mode=vision.RunningMode.VIDEO, num_hands=2
    )
    hand_detector = vision.HandLandmarker.create_from_options(hand_options)

    time_prev = time.time()
    frame_counter = 0
    last_timestamp_ms = 0

    feature_extractor_thread = ThreadPoolExecutor(max_workers=3)

    FRAMES_PATH = "./test_9_16_FHD/"
    frames_path = Path(FRAMES_PATH)

    frames = []
    for frame_path in sorted(frames_path.rglob("*.jpg")):
        frame = cv2.imread(frame_path)
        frames.append(frame)

    for frame in frames:
        # time.sleep(1)
        frame_counter += 1

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        timestamp_ms = int(time.time() * 1000)
        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1
        last_timestamp_ms = timestamp_ms

        future_face = feature_extractor_thread.submit(
            face_detector.detect_for_video, mp_image, timestamp_ms
        )
        future_pose = feature_extractor_thread.submit(
            pose_detector.detect_for_video, mp_image, timestamp_ms
        )
        future_hands = feature_extractor_thread.submit(
            hand_detector.detect_for_video, mp_image, timestamp_ms
        )

        face_result = future_face.result()
        pose_result = future_pose.result()
        hand_result = future_hands.result()

        annotated_image = np.copy(mp_image.numpy_view())

        annotated_image = draw_face_landmarks_on_image(annotated_image, face_result)
        annotated_image = draw_pose_landmarks_on_image(annotated_image, pose_result)
        annotated_image = draw_hand_landmarks_on_image(annotated_image, hand_result)

        bgr_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        time_now = time.time()
        time_diff = time_now - time_prev
        if time_diff > 1:
            print(f"FPS: {frame_counter}")
            frame_counter = 0
            time_prev = time.time()

        if not show_frames:
            continue

        cv2.imshow("mediapipe benchmark", bgr_annotated_image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    face_detector.close()
    pose_detector.close()
    hand_detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    synchronous_detect_benchmark(show_frames=False)
