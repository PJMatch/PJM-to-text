"""Module for extraction from PJM dataset."""

import numpy as np
import cv2
from pathlib import Path
import time

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from extraction_4D_1 import extract_raw_keypoints


TASKS_DIR = Path("../mediapipe_tasks")
FACE_MODEL_PATH = TASKS_DIR / "face_landmarker_v2_with_blendshapes.task"
POSE_MODEL_PATH = TASKS_DIR / "pose_landmarker_lite.task"
HAND_MODEL_PATH = TASKS_DIR / "hand_landmarker.task"

DATASET_PATH = Path("/pjm/baza_wideo")
OUTPUT_PATH = Path("/pjm/extracted")

DATASET_FPS = 30

def extract_frames(path: str):
    """Extract frames from a video."""
    video = cv2.VideoCapture(str(path))

    n = 0
    try:
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            yield n, frame
            n += 1
    finally:
        video.release()


def get_files_to_process(processed: set) -> set:
    """Returns files that have not been processed yet."""
    files_mp4 = set()
    for file_mp4 in DATASET_PATH.rglob("*.[mM][pP]4"):
        files_mp4.add(file_mp4)
    
    return files_mp4 - processed

def init_mediapipe() -> dict:
    pose_detector = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO)
    )

    hand_detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2)
    )
    
    face_detector = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1)
    )

    return {
        "pose": pose_detector,
        "hands": hand_detector,
        "face": face_detector
    }

def get_processed_filenames() -> set:
    """Reads from the log and returns a set (for O(1) lookup time) with processed files."""
    processed_log = OUTPUT_PATH / "processed_log.txt"

    if not processed_log.exists():
        print(f"File {str(processed_log)} does NOT exist")
        return set()

    with open(processed_log, 'r') as file:
        processed_set = {line.strip() for line in file}
    
    return processed_set

def run_mp_inference(detectors: dict, frame, timestamp_ms):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    pose_result = detectors["pose"].detect_for_video(mp_image, timestamp_ms) 
    hands_result = detectors["hands"].detect_for_video(mp_image, timestamp_ms)
    face_result = detectors["face"].detect_for_video(mp_image, timestamp_ms)

    return pose_result, hands_result, face_result

def process_sequence(pjm_file: Path, video_fps, detectors):
    """Extract keypoints from file (sequence) and return keypoints sequence."""

    sequence_data = []
    sequence_name = pjm_file.stem

    for frame_id, frame in extract_frames(pjm_file):
        timestamp_ms = int(frame_id * 1000 / video_fps)

        pose_res, hands_res, face_res = run_mp_inference(detectors, frame, timestamp_ms)
        raw_keypoints = extract_raw_keypoints(pose_res, hands_res, face_res) 
        sequence_data.append(raw_keypoints)

    if not sequence_data:
        return None, None

    return sequence_data, sequence_name

def process_file(pjm_file, detectors):
    sequence_data, sequence_name = process_sequence(pjm_file, DATASET_FPS, detectors)
    if sequence_data is None:
        print(f"Sequence data is empty in {str(pjm_file)}")
        return False
    np.save(OUTPUT_PATH / f"{sequence_name}.npy", np.array(sequence_data, dtype=object))

    processed_log = OUTPUT_PATH / "processed_log.txt"
    with open(processed_log, "a", encoding="utf-8") as f:
        f.write(f"{str(pjm_file)}\n")
        print(f"Successfully processed file {str(pjm_file)}")
    
    return True


def process_pjm():
    processed = get_processed_filenames()    
    files_to_process = get_files_to_process(processed)

    err_file_set = set()
    detectors = init_mediapipe()
    for pjm_file in files_to_process:
        try:
            if not process_file(pjm_file, detectors):
                err_file_set.add(str(pjm_file))
                continue
        except Exception as e:
            print(f"ERROR processing {pjm_file}: {e}")
            err_file_set.add(str(pjm_file))
            continue

        break
    
    return err_file_set

def main():
    res_err = process_pjm()
    if res_err:
        print(f"Error occured in files {res_err}")
    else:
        print("Succesfully extracted features from all videos")

if __name__ == "__main__":
    main()