"""Module for frames extraction from a video."""

import cv2
from pathlib import Path

from extraction_4D_1 import extract_raw_keypoints


TASKS_DIR = Path("../mediapipe_tasks")
FACE_MODEL_PATH = TASKS_DIR / "face_landmarker_v2_with_blendshapes.task"
POSE_MODEL_PATH = TASKS_DIR / "pose_landmarker_lite.task"
HAND_MODEL_PATH = TASKS_DIR / "hand_landmarker.task"

DATASET_PATH = Path("/pjm/baza_wideo")
OUTPUT_PATH = Path("/pjm/extracted")

def extract_frames(path: str) -> list:
    """Extract frames from a video."""
    video = cv2.VideoCapture(path)

    frames = []
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        frames.append(frame)

    return frames


def get_files_to_process(processed: set):
    """Returns files that have not been processed yet."""
    files_mp4 = set()
    for file_mp4 in DATASET_PATH.rglob("*.[mM][pP]4"):
        files_mp4.add(file_mp4)
    
    return files_mp4 - processed

def init_mediapipe():
    pose_detector = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO)
    )

    hand_detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2)
    )
    
    face_detector = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1)
    )

    return pose_detector, hand_detector, face_detector

def get_processed_filenames() -> set:
    """Reads from the log and returns a set (for O(1) lookup time) with processed files."""
    processed_log = OUTPUT_PATH / "processed_log.txt"

    if not processed_log.exists():
        print(f"File {str(processed_log)} does NOT exist")
        return set()

    with open(processed_log, 'r') as file:
        processed_set = {line.strip() for line in file}
    
    return processed_set

def process_file(pjm_file: Path):
    """Extract keypoints from file and return keypoints sequence."""
    frames = extract_frames(pjm_file)

    if len(frames) == 0:
        return pjm_file

    sequence_data = []
    sequence_name = pjm_file.stem
    for frame in frames:
        raw_keypoints = extract_raw_keypoints() 
        sequence_data.append(raw_keypoints)
    return sequence_data, sequence_name

def process_pjm():
    processed = get_processed_filenames()    
    files_to_process = get_files_to_process(processed)

    frames = []
    for pjm_file in files_to_process:
        sequence_data, sequence_name = process_file(pjm_file)
        np.save(OUTPUT_PATH / f"{sequence_name}.npy", np.array(sequence_data, dtype=object))

        processed_log = OUTPUT_PATH / "processed_log.txt"
        with open(processed_log, "a", encoding="utf-8") as f:
            f.write(f"{str(pjm_file)}\n")
        
        print(f"Successfully processed file {str(pjm_file)}")
        break

    return None
        

def main():
    res_err = process_pjm()
    if res != None:
        print(f"Error occured in file {res_err}")
    else:
        print("Succesfully extracted features from all videos")

if __name__ == "__main__":
    process_pjm()