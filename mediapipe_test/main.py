# most of the code has been copied from the official mediapipe website and later only finetuded
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

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time


def draw_face_landmarks_on_image(rgb_image, detection_result):
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
            connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style())
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style())
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())

    return annotated_image

def draw_pose_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # filter out all connections that start or end under 10 (face points)
    body_only_connections = [
        conn for conn in vision.PoseLandmarksConnections.POSE_LANDMARKS
        if conn.start > 10 and conn.end > 10
    ]

    pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
    pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

    class InvisibleLandmark:
        visibility = 0.0

    for pose_landmarks in pose_landmarks_list:
        
        filtered_landmarks = [
            InvisibleLandmark() if i <= 10 else lm 
            for i, lm in enumerate(pose_landmarks)
        ]

        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=filtered_landmarks,
            connections=body_only_connections,
            landmark_drawing_spec=pose_landmark_style,
            connection_drawing_spec=pose_connection_style)

    return annotated_image

def plot_face_blendshapes_bar_graph(face_blendshapes):
    # Extract the face blendshapes category names and scores.
    face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
    face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
    # The blendshapes are ordered in decreasing score value.
    face_blendshapes_ranks = range(len(face_blendshapes_names))

    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()

    # Label each bar with values
    for score, patch in zip(face_blendshapes_scores, bar.patches):
        plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

    ax.set_xlabel('Score')
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    plt.show()

latest_face_result = None
latest_pose_result = None

def face_result_callback(result, output_image, timestamp_ms):
    global latest_face_result
    latest_face_result = result

def pose_result_callback(result, output_image, timestamp_ms):
    global latest_pose_result
    latest_pose_result = result

if __name__ == "__main__":
    
    # face mesh detector
    face_base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
    face_options = vision.FaceLandmarkerOptions(
        base_options=face_base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=face_result_callback,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1)
    face_detector = vision.FaceLandmarker.create_from_options(face_options)

    # pose detector
    pose_base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
    pose_options = vision.PoseLandmarkerOptions(
        base_options=pose_base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=pose_result_callback,
        output_segmentation_masks=False)
    pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

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

        annotated_image = np.copy(image.numpy_view())
        
        if latest_face_result is not None and latest_face_result.face_landmarks:
            annotated_image = draw_face_landmarks_on_image(annotated_image, latest_face_result)
            
        if latest_pose_result is not None and latest_pose_result.pose_landmarks:
            annotated_image = draw_pose_landmarks_on_image(annotated_image, latest_pose_result)

        rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        cv2.imshow('mediapipe test', rgb_annotated_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time_now = time.time()
        time_diff = time_now - time_prev
        if time_diff > 1:
            print(f'FPS: {frame_counter}')
            frame_counter = 0
            time_prev = time.time()

    face_detector.close()
    pose_detector.close()
    camera.release()
    cv2.destroyAllWindows()