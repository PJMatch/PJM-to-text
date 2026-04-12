"""Module for extraction from PJM dataset."""

import time
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from extraction_4D_1 import extract_raw_keypoints
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

TASKS_DIR = Path("../mediapipe_tasks")
FACE_MODEL_PATH = TASKS_DIR / "face_landmarker_v2_with_blendshapes.task"
POSE_MODEL_PATH = TASKS_DIR / "pose_landmarker_lite.task"
HAND_MODEL_PATH = TASKS_DIR / "hand_landmarker.task"

DATASET_PATH = Path("/pjm/baza_wideo")
OUTPUT_PATH = Path("/pjm/extracted")

DATASET_FPS = 30

FEATURE_EXTRACTOR_THREAD = ThreadPoolExecutor(max_workers=3)


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


def is_recorded_correctly(file_path: Path):
    """Returns True if the file has been maked as recorded correctly"""
    path_str = str(file_path)
    no_extention = path_str[: path_str.find(".")]
    json_path = str(no_extention) + ".json"
    with open(json_path, "r") as json_file:
        data = json.load(json_file)
        return data["recorded_correctly"]


def get_files_to_process(processed: set) -> set:
    """Returns files that have not been processed yet."""
    files_mp4 = set()
    for file_mp4 in DATASET_PATH.rglob("*.[mM][pP]4"):
        if not is_recorded_correctly(file_mp4):
            continue
        files_mp4.add(file_mp4)

    return files_mp4 - processed


def load_models_to_memory() -> dict:
    """Reads the .task files from disk into RAM."""
    with open(POSE_MODEL_PATH, "rb") as f:
        pose_bytes = f.read()
    with open(HAND_MODEL_PATH, "rb") as f:
        hand_bytes = f.read()
    with open(FACE_MODEL_PATH, "rb") as f:
        face_bytes = f.read()

    return {"pose": pose_bytes, "hands": hand_bytes, "face": face_bytes}


def init_mediapipe(models_buffer: dict) -> dict:
    """Initialize mediapipe models from RAM buffer.

    Args:
        models_buffer(dict): contains preloaded models in byte form
    Returns:
        dict: dictionary with initialized models' instances
    """
    pose_detector = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_buffer=models_buffer["pose"]),
            running_mode=vision.RunningMode.VIDEO,
        )
    )

    hand_detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_buffer=models_buffer["hands"]),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
        )
    )

    face_detector = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_buffer=models_buffer["face"]),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
        )
    )

    return {"pose": pose_detector, "hands": hand_detector, "face": face_detector}


def get_processed_filenames() -> set:
    """Reads from the log and returns a set (for O(1) lookup time) with processed files."""
    processed_log = OUTPUT_PATH / "processed_log.txt"

    if not processed_log.exists():
        print(f"File {str(processed_log)} does NOT exist")
        return set()

    with open(processed_log, "r") as file:
        processed_set = {line.strip() for line in file}

    return processed_set


def run_mp_inference_optim(detectors: dict, frame, timestamp_ms):
    """Runs MediaPipe inferecne with multithreading."""
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )

    pose = FEATURE_EXTRACTOR_THREAD.submit(
        detectors["pose"].detect_for_video, mp_image, timestamp_ms
    )
    hands = FEATURE_EXTRACTOR_THREAD.submit(
        detectors["hands"].detect_for_video, mp_image, timestamp_ms
    )
    face = FEATURE_EXTRACTOR_THREAD.submit(
        detectors["face"].detect_for_video, mp_image, timestamp_ms
    )

    face_result = face.result()
    pose_result = pose.result()
    hands_result = hands.result()

    return pose_result, hands_result, face_result


def run_mp_inference(detectors: dict, frame, timestamp_ms: int):
    """Runs MediaPipe inference.

    Args:
        detectors(dict): dictionary containing loaded mediapipe models
        frame: frame to process
        timestamp_ms(int): MediaPipe timestamp
    """
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )

    pose_result = detectors["pose"].detect_for_video(mp_image, timestamp_ms)
    hands_result = detectors["hands"].detect_for_video(mp_image, timestamp_ms)
    face_result = detectors["face"].detect_for_video(mp_image, timestamp_ms)

    return pose_result, hands_result, face_result


def process_sequence(pjm_file: Path, video_fps, detectors):
    """Extract keypoints from file (sequence) and return keypoints sequence."""
    sequence_data = []
    sequence_name = pjm_file.stem

    time_prev = time.time()
    frame_counter = 0
    for frame_id, frame in extract_frames(pjm_file):
        timestamp_ms = int(frame_id * 1000 / video_fps)

        pose_res, hands_res, face_res = run_mp_inference_optim(detectors, frame, timestamp_ms)
        raw_keypoints = extract_raw_keypoints(pose_res, hands_res, face_res)
        sequence_data.append(raw_keypoints)

        frame_counter += 1
        time_now = time.time()
        time_diff = time_now - time_prev

        if time_diff > 1:
            # print(f"FPS: {frame_counter}")
            frame_counter = 0
            time_prev = time.time()

    if not sequence_data:
        return None, None

    return sequence_data, sequence_name


def process_file(pjm_file, detectors):
    """Processes file with MediaPipe detectors.

    Returns:
        bool: False if sequence data is empty, else True
    """
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
    """Process PJM dataset.

    Returns:
        set: set containing filenames in which process finished with an error
    """
    processed = get_processed_filenames()
    files_to_process = get_files_to_process(processed)

    models_buffer = load_models_to_memory()

    err_file_set = set()

    for i, pjm_file in enumerate(files_to_process):
        try:
            # We need to init models  for every file because in VIDEO mode the models require
            # timestamp (which is assigned per the model instance) to always be monotonous.
            # That makes zeroing the timestamp at the beginning of every file an Error.
            # Global timestamp also is considered an error since the model will try to
            # interpolate between files since it would think that it is still looking at the same
            # file.
            # To save time on reloading the MP models we first loaded the models to RAM
            # from which they are then reinitialized instead of reaching to the hard disk every time
            detectors = init_mediapipe(models_buffer)

            if not process_file(pjm_file, detectors):
                err_file_set.add(str(pjm_file))
                continue
        except Exception as e:
            print(f"ERROR processing {pjm_file}: {e}")
            err_file_set.add(str(pjm_file))
            continue

        if i == 2:
            break

    return err_file_set


def main():
    """Main function."""
    res_err = process_pjm()
    if res_err:
        print(f"Error occured in files {res_err}")
    else:
        print("Succesfully extracted features from all videos")


if __name__ == "__main__":
    # main()
    print(is_recorded_correctly("/pjm/baza_wideo/17_185_20260401_173103.mp4"))
