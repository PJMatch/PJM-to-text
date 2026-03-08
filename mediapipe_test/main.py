"""Mediapipe keypoints detection module.

This module provides functions that allow detection and display of detected keypoints
using Mediapipe using it's Tasks API. The implementation tries to imitate the Mediapipe Holistic
which at the time is not available in the new API.
"""
# most of the code has been copied from the official mediapipe website and later only finetuned
# for our usecase, source: https://ai.google.dev/edge/mediapipe/solutions/guide

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

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils


def draw_face_landmarks_on_image(rgb_image: np.ndarray, detection_result):
    """Draw face landmarks on image.

    Args:
        rgb_image (np.ndarray): image to draw onto,
        detection_result: result of mediapipe detection
    Returns:
        nd.array: resulting annotated image
    """
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
        )
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
        )
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
        )

    return annotated_image


def draw_pose_landmarks_on_image(rgb_image, detection_result):
    """Draw pose landmarks on image.

    Args:
        rgb_image (np.ndarray): image to draw onto,
        detection_result: result of mediapipe detection
    Returns:
        nd.array: resulting annotated image
    """
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # filter out all connections that start or end under 10 (face points)
    body_only_connections = [
        conn
        for conn in vision.PoseLandmarksConnections.POSE_LANDMARKS
        if conn.start > 10 and conn.end > 10
    ]

    pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
    pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

    class InvisibleLandmark:
        visibility = 0.0

    for pose_landmarks in pose_landmarks_list:
        filtered_landmarks = [
            InvisibleLandmark() if i <= 10 else lm for i, lm in enumerate(pose_landmarks)
        ]

        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=filtered_landmarks,
            connections=body_only_connections,
            landmark_drawing_spec=pose_landmark_style,
            connection_drawing_spec=pose_connection_style,
        )

    return annotated_image


mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


def draw_hand_landmarks_on_image(rgb_image, detection_result):
    """Draw hand landmarks on image.

    Args:
        rgb_image (np.ndarray): image to draw onto,
        detection_result: result of mediapipe detection
    Returns:
        nd.array: resulting annotated image
    """
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(
            annotated_image,
            f"{handedness[0].category_name}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            HANDEDNESS_TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return annotated_image


def face_result_callback(result, output_image, timestamp_ms) -> None:
    """Updates global with the most recent detection from Mediapipe.

    This function acts as a callback for the MediaPipe Face Landmarker
    running in live stream mode.

    Args:
        result: The FaceLandmarkerResult object containing detected face landmarks.
        output_image: The specific image frame that was used for detection.
        timestamp_ms: The hardware display timestamp of the input image in milliseconds.

    Returns:
        None
    """
    global latest_face_result
    latest_face_result = result


def pose_result_callback(result, output_image, timestamp_ms):
    """Updates global with the most recent detection from Mediapipe.

    This function acts as a callback for the MediaPipe Face Landmarker
    running in live stream mode.

    Args:
        result: The PoseLandmarkerResult object containing detected face landmarks.
        output_image: The specific image frame that was used for detection.
        timestamp_ms: The hardware display timestamp of the input image in milliseconds.

    Returns:
        None
    """
    global latest_pose_result
    latest_pose_result = result


def hand_result_callback(result, output_image, timestamp_ms):
    """Updates global with the most recent detection from Mediapipe.

    This function acts as a callback for the MediaPipe Face Landmarker
    running in live stream mode.

    Args:
        result: The HandLandmarkerResult object containing detected face landmarks.
        output_image: The specific image frame that was used for detection.
        timestamp_ms: The hardware display timestamp of the input image in milliseconds.

    Returns:
        None
    """
    global latest_hand_result
    latest_hand_result = result


latest_face_result = None
latest_pose_result = None
latest_hand_result = None


def asynchronous_detect() -> None:
    """Asynchronously detects landmarks and draws them on device camera livestream.

    This function provides a non-blocking version of detection that displays detected
    keypoints on the live camera feedback. If detection for one frame takes more time then
    it takes for another frame to come, it skips the incoming frame.

    Returns:
        None
    """
    # face mesh detector
    face_base_options = python.BaseOptions(
        model_asset_path="face_landmarker_v2_with_blendshapes.task"
    )
    face_options = vision.FaceLandmarkerOptions(
        base_options=face_base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=face_result_callback,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    face_detector = vision.FaceLandmarker.create_from_options(face_options)

    # pose detector
    pose_base_options = python.BaseOptions(model_asset_path="pose_landmarker_lite.task")
    pose_options = vision.PoseLandmarkerOptions(
        base_options=pose_base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=pose_result_callback,
        output_segmentation_masks=False,
    )
    pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

    # hand landmakr detector
    hand_base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
    hand_options = vision.HandLandmarkerOptions(
        base_options=hand_base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=hand_result_callback,
        num_hands=2,
    )
    hand_detector = vision.HandLandmarker.create_from_options(hand_options)

    camera = cv2.VideoCapture(0)

    time_prev = time.time()
    frame_counter = 0
    last_timestamp_ms = 0

    while camera.isOpened():
        frame_counter += 1
        ret, frame = camera.read()

        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        timestamp_ms = int(time.time() * 1000)
        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1
        last_timestamp_ms = timestamp_ms

        face_detector.detect_async(image, timestamp_ms)
        pose_detector.detect_async(image, timestamp_ms)
        hand_detector.detect_async(image, timestamp_ms)

        annotated_image = np.copy(image.numpy_view())

        if latest_face_result is not None and latest_face_result.face_landmarks:
            annotated_image = draw_face_landmarks_on_image(annotated_image, latest_face_result)

        if latest_pose_result is not None and latest_pose_result.pose_landmarks:
            annotated_image = draw_pose_landmarks_on_image(annotated_image, latest_pose_result)

        if latest_hand_result is not None and latest_hand_result.hand_landmarks:
            annotated_image = draw_hand_landmarks_on_image(annotated_image, latest_hand_result)

        rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        cv2.imshow("mediapipe test", rgb_annotated_image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        time_now = time.time()
        time_diff = time_now - time_prev
        if time_diff > 1:
            print(f"FPS: {frame_counter}")
            frame_counter = 0
            time_prev = time.time()

    face_detector.close()
    pose_detector.close()
    camera.release()
    cv2.destroyAllWindows()


def synchronous_detect():
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

    camera = cv2.VideoCapture(0)

    time_prev = time.time()
    frame_counter = 0
    last_timestamp_ms = 0

    while camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            break

        frame_counter += 1

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        timestamp_ms = int(time.time() * 1000)
        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1
        last_timestamp_ms = timestamp_ms

        face_result = face_detector.detect_for_video(mp_image, timestamp_ms)
        pose_result = pose_detector.detect_for_video(mp_image, timestamp_ms)
        hand_result = hand_detector.detect_for_video(mp_image, timestamp_ms)

        annotated_image = np.copy(mp_image.numpy_view())

        if face_result and face_result.face_landmarks:
            annotated_image = draw_face_landmarks_on_image(annotated_image, face_result)

        if pose_result and pose_result.pose_landmarks:
            annotated_image = draw_pose_landmarks_on_image(annotated_image, pose_result)

        if hand_result and hand_result.hand_landmarks:
            annotated_image = draw_hand_landmarks_on_image(annotated_image, hand_result)

        bgr_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        cv2.imshow("mediapipe test", bgr_annotated_image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        time_now = time.time()
        time_diff = time_now - time_prev
        if time_diff > 1:
            print(f"FPS: {frame_counter}")
            frame_counter = 0
            time_prev = time.time()

    face_detector.close()
    pose_detector.close()
    hand_detector.close()
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    synchronous_detect()
