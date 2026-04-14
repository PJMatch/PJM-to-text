import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# Paths
PHOENIX_ROOT = Path(r"input/dataset")
OUTPUT_ROOT = Path(r"output/dataset")
FACE_MODEL_PATH = "face_landmarker_v2_with_blendshapes.task"


def extract_raw_keypoints(pose_result, hand_result, face_result):
    """
    Extracts visible landmarks into a structured dictionary without zero-padding.
    Confidence score is extracted from pose visibility.
    Face and hands default to 1.0 if detected.
    Missing components remain as empty lists.
    """
    frame_data = {"pose": [], "face": [], "lh": [], "rh": []}

    # Pose (33 points)
    if pose_result and pose_result.pose_landmarks:
        frame_data["pose"] = [
            [lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0)]  # meidapipe native visibility score
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


def main():
    """
    Main loop for RAW extraction.
    Saves sequences as arrays of dictionaries.
    """
    pose_detector = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path="pose_landmarker_lite.task"),
            running_mode=vision.RunningMode.VIDEO,
        )
    )

    hand_detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path="hand_landmarker.task"),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
        )
    )

    face_detector = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
        )
    )

    dataset_splits = ["train", "dev", "test"]
    global_timestamp_ms = 0

    for split in dataset_splits:
        split_dir = PHOENIX_ROOT / split
        output_split_dir = OUTPUT_ROOT / split

        if not split_dir.exists():
            continue

        output_split_dir.mkdir(parents=True, exist_ok=True)
        subfolders = [sf for sf in split_dir.iterdir() if sf.is_dir()]

        for folder in tqdm(subfolders, desc=f"Processing RAW {split.upper()} set"):
            seq_name = folder.name
            sequence_data = []

            image_files = sorted(folder.glob("*.png"))
            if not image_files:
                continue

            for img_path in image_files:
                frame = cv2.imread(str(img_path))
                if frame is None:
                    continue

                # BGR to RGB conversion
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )

                # Inference
                pose_result = pose_detector.detect_for_video(mp_image, global_timestamp_ms)
                hand_result = hand_detector.detect_for_video(mp_image, global_timestamp_ms)
                face_result = face_detector.detect_for_video(mp_image, global_timestamp_ms)

                # Extraction
                raw_keypoints = extract_raw_keypoints(pose_result, hand_result, face_result)
                sequence_data.append(raw_keypoints)

                # integer incrementign
                global_timestamp_ms += int(33)

            # Save as npy
            np.save(output_split_dir / f"{seq_name}.npy", np.array(sequence_data, dtype=object))

    pose_detector.close()
    hand_detector.close()
    face_detector.close()
    print("\nRAW dataset extraction completed successfully.")


if __name__ == "__main__":
    main()
