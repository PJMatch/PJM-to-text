import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import json
from pathlib import Path
from tqdm import tqdm

#Paths
PHOENIX_ROOT = Path(r'input/dataset') 
OUTPUT_ROOT = Path(r'output/dataset')
FACE_MODEL_PATH = 'face_landmarker_v2_with_blendshapes.task'

def extract_and_pad_keypoints(pose_result, hand_result, face_result):
    """
    Extracts visible landmarks and applies zero-padding directly in one go.
    Returns a flat list of exactly 2212 floats per frame.
    Missing components are padded with 0.0 (including the confidence score).
    """
    #zero padded list for each 
    pose_data = [0.0] * (33 * 4)
    face_data = [0.0] * (478 * 4)
    lh_data = [0.0] * (21 * 4)
    rh_data = [0.0] * (21 * 4)

    #Pose
    if pose_result and pose_result.pose_landmarks:
        pose_coords = []
        for lm in pose_result.pose_landmarks[0]:
            # mediapipe native visibility score
            pose_coords.extend([float(lm.x), float(lm.y), float(lm.z), float(getattr(lm, 'visibility', 1.0))])
        pose_data = pose_coords

    #Face
    if face_result and face_result.face_landmarks:
        face_coords = []
        for lm in face_result.face_landmarks[0]:
            # confidence score set to 1.0 by default
            face_coords.extend([float(lm.x), float(lm.y), float(lm.z), 1.0])
        face_data = face_coords[:1912] # cutoff for limit

    #Hands
    if hand_result and hand_result.hand_landmarks:
        for idx, hand_landmarks in enumerate(hand_result.hand_landmarks):
            handedness = hand_result.handedness[idx][0].category_name
            hand_coords = []
            for lm in hand_landmarks:
                #confidence score set to 1.0 by default
                hand_coords.extend([float(lm.x), float(lm.y), float(lm.z), 1.0])
            
            if handedness == 'Left':
                lh_data = hand_coords
            elif handedness == 'Right':
                rh_data = hand_coords

    #Concatenate into flat list of 2212 floats
    return pose_data + face_data + lh_data + rh_data

def main():
    """
    Main loop for extraction and padding. 
    Saves sequences directly as padded .json files.
    """
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

    dataset_splits = ['train', 'dev', 'test']
    global_timestamp_ms = 0

    for split in dataset_splits:
        split_dir = PHOENIX_ROOT / split
        output_split_dir = OUTPUT_ROOT / split
        
        if not split_dir.exists():
            continue

        output_split_dir.mkdir(parents=True, exist_ok=True)
        subfolders = [sf for sf in split_dir.iterdir() if sf.is_dir()]

        for folder in tqdm(subfolders, desc=f"Processing {split.upper()} set"):
            seq_name = folder.name
            sequence_data = []
            
            image_files = sorted(folder.glob('*.png'))
            if not image_files:
                continue

            for img_path in image_files:
                frame = cv2.imread(str(img_path))
                if frame is None:
                    continue

                #BGR to RGB conversion
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                #Inference
                pose_result = pose_detector.detect_for_video(mp_image, global_timestamp_ms) 
                hand_result = hand_detector.detect_for_video(mp_image, global_timestamp_ms)
                face_result = face_detector.detect_for_video(mp_image, global_timestamp_ms)

                #Extraction with padding
                padded_keypoints = extract_and_pad_keypoints(pose_result, hand_result, face_result)
                sequence_data.append(padded_keypoints)

                #integer incrementign
                global_timestamp_ms += int(33)

            #Save as JSON
            json_file_path = output_split_dir / f"{seq_name}.json"
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(sequence_data, f)

    pose_detector.close()
    hand_detector.close()
    face_detector.close()
    print("\nDataset extraction and padding to JSON completed successfully.")

if __name__ == "__main__":
    main()