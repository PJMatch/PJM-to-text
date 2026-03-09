import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# Sample was done in branch
PHOENIX_PATH = Path('phoenix_sample') 
OUTPUT_DIR = Path('sample_results')
FACE_MODEL_PATH = 'face_landmarker_v2_with_blendshapes.task'

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
        # Ensure strict length adherence to prevent shape errors
        if len(coords) <= 1434:
            face_data[:len(coords)] = coords
        else:
            face_data = coords[:1434]

    # Hands = 67 indices (times two hands) = 126
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

def main():
    """
    Main execution loop: Iterates through subdirectories, processes frames, 
    and saves the extracted keypoints as NumPy arrays.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize MediaPipe
    pose_detector = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path='pose_landmarker_lite.task'),
            running_mode=vision.RunningMode.VIDEO)
    )

    hand_detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2)
    )
    
    face_detector = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1)
    )

    # Identify subdirectories with sequences
    subfolders = [sf for sf in PHOENIX_PATH.iterdir() if sf.is_dir()]

    # Global timestamp outside the loops to ensure monotonic increase
    global_timestamp_ms = 0

    for folder in tqdm(subfolders, desc="Processing Phoenix sequences"):
        seq_name = folder.name
        sequence_data = []
        
        # Sorting
        image_files = sorted(folder.glob('*.png'))
        
        if not image_files:
            continue

        for img_path in image_files:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            # Conversion BGR->RGB
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Inference
            pose_result = pose_detector.detect_for_video(mp_image, global_timestamp_ms) 
            hand_result = hand_detector.detect_for_video(mp_image, global_timestamp_ms)
            face_result = face_detector.detect_for_video(mp_image, global_timestamp_ms)

            # Feature extraction with zer-padding
            keypoints = extract_keypoints(pose_result, hand_result, face_result)
            sequence_data.append(keypoints)

            # Increment the global timestamp (40 ms because of the 25 FPS)
            global_timestamp_ms += 40

        # Sequence saved as a binary NumPy file
        np.save(OUTPUT_DIR / f"{seq_name}.npy", np.array(sequence_data))

    pose_detector.close()
    hand_detector.close()
    face_detector.close()
    print("\nPreprocessing sequence completed successfully.")

if __name__ == "__main__":
    main()