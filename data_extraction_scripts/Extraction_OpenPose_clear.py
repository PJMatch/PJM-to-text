"""
OpenPose to CoSign (ST-GCN) Pipeline.

- Open Pose: https://github.com/cmu-perceptual-computing-lab/openpose/releases
(if model fails to download, use the following links and place them in the 
models/ directory of OpenPose)
- Face: https://www.dropbox.com/s/d08srojpvwnk252/pose_iter_116000.caffemodel?dl=1
- Hand: https://www.dropbox.com/s/gqgsme6sgoo0zxf/pose_iter_102000.caffemodel?dl=1
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from typing import List

import numpy as np


def run_openpose_on_images(
    openpose_bin: str, openpose_root: str, image_dir: str, temp_json_dir: str
) -> None:
    """Runs the OpenPose executable on a directory of images to extract keypoints."""
    os.makedirs(temp_json_dir, exist_ok=True)

    if not os.path.exists(openpose_bin):
        print(f"Error: OpenPose executable not found at {openpose_bin}")
        sys.exit(1)

    command = [
        openpose_bin,
        "--image_dir",
        image_dir,
        "--write_json",
        temp_json_dir,
        "--hand",
        "--face",
        "--display",
        "0",
        "--render_pose",
        "0",
        "--number_people_max",
        "1",
        "--net_resolution",
        "-1x256",
    ]

    subprocess.run(command, check=True, cwd=openpose_root)


def convert_jsons_to_cosign_tensor(json_dir: str, output_npy_path: str) -> None:
    """Converts OpenPose JSON output files into a formatted CoSign tensor."""
    json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    if not json_files:
        return

    num_frames = len(json_files)
    video_data = np.zeros((num_frames, 77, 3), dtype=np.float32)

    for t, file_path in enumerate(json_files):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not data.get("people"):
            continue

        person = data["people"][0]

        body_raw = np.array(person["pose_keypoints_2d"]).reshape(-1, 3)
        body_pts = body_raw[[0, 1, 2, 3, 4, 5, 6, 7, 8]]

        left_hand_pts = np.array(person["hand_left_keypoints_2d"]).reshape(21, 3)
        right_hand_pts = np.array(person["hand_right_keypoints_2d"]).reshape(21, 3)

        face_raw = np.array(person["face_keypoints_2d"]).reshape(70, 3)
        mouth_pts = face_raw[list(range(60, 68))]
        face_pts = face_raw[
            [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 36, 37, 38, 39, 42, 43, 44, 45]
        ]

        video_data[t] = np.concatenate(
            [body_pts, left_hand_pts, right_hand_pts, mouth_pts, face_pts], axis=0
        )

    cosign_tensor = np.transpose(video_data, (2, 0, 1))
    cosign_tensor = np.expand_dims(cosign_tensor, axis=-1)

    np.save(output_npy_path, cosign_tensor)


def process_dataset_folders(openpose_bin: str, openpose_root: str, base_folders: List[str]) -> None:
    """Processes dataset folders through the OpenPose and CoSign conversion pipeline."""
    temp_workspace = os.path.join(os.getcwd(), "temp_openpose_jsons")

    for base_folder in base_folders:
        if not os.path.isdir(base_folder):
            continue

        print(f"Processing split: {os.path.basename(base_folder)}")
        subdirs = [f.path for f in os.scandir(base_folder) if f.is_dir()]

        for seq_dir in subdirs:
            seq_name = os.path.basename(seq_dir)
            output_npy_path = os.path.join(seq_dir, f"{seq_name}.npy")

            if not glob.glob(os.path.join(seq_dir, "*.png")):
                continue

            run_openpose_on_images(openpose_bin, openpose_root, seq_dir, temp_workspace)
            convert_jsons_to_cosign_tensor(temp_workspace, output_npy_path)

            if os.path.exists(temp_workspace):
                shutil.rmtree(temp_workspace)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenPose and convert to CoSign format.")
    parser.add_argument(
        "--openpose_root",
        type=str,
        required=True,
        help="Path to OpenPose root directory (containing models/)",
    )
    parser.add_argument(
        "--openpose_bin", type=str, required=True, help="Path to OpenPose executable (.exe or .bin)"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to dataset containing train/dev/test directories",
    )

    args = parser.parse_args()

    target_folders = [
        os.path.join(args.dataset_dir, "train"),
        os.path.join(args.dataset_dir, "dev"),
        os.path.join(args.dataset_dir, "test"),
    ]

    print("Start")
    process_dataset_folders(args.openpose_bin, args.openpose_root, target_folders)
    print("Success")
    # I used: src/Extraction_OpenPose_clear.py 
    # --openpose_root "C:\openpose" --openpose_bin "C:\openpose\bin\OpenPoseDemo.exe"
    # --dataset_dir "C:\PJM_projekt\PJM-to-text\src"
    # U need to adjust the paths according to your setup.
