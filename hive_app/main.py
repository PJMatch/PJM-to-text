import argparse
import io
import os
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import requests

from mediapipe_process import process_video

SERVER = "http://34.118.102.249:8000"
ANNOTATIONS_OUTPUT = "keypoints.zip"
DOWNLOAD_CHUNK_SIZE = 1024 * 1024

def send_result_with_retry(server: str, task_code: str, buf: io.BytesIO, retries: int = 3) -> None:
    url = f"{server}/finish-task/{task_code}"
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            buf.seek(0)
            response = requests.post(
                url,
                files={"file": ("result.npy", buf, "application/octet-stream")},
                timeout=1200,
            )
            response.raise_for_status()
            return
        except requests.RequestException as exc:
            last_error = exc
            print(f"error upload fail {attempt}/{retries} for {task_code}: {exc}")
            if attempt < retries:
                time.sleep(2 * attempt)

    raise RuntimeError(f"error upload failed - task {task_code}: {last_error}")


def download_file_with_progress(url: str, output_path: str, timeout: int) -> None:
    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", "0"))
    downloaded = 0
    last_percent = -1

    with open(output_path, "wb") as output_file:
        for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
            if not chunk:
                continue

            output_file.write(chunk)
            downloaded += len(chunk)

            if total_size > 0:
                percent = int((downloaded * 100) / total_size)
                if percent >= last_percent + 5 or percent == 100:
                    print(f"Downloading: {percent}% ({downloaded}/{total_size} bytes)", end="\r", flush=True)
                    last_percent = percent

    if total_size > 0:
        print("Downloading: 100%", " " * 30)


def build_process_progress_callback() -> Callable[[int, int], None]:
    last_percent = -1
    last_count_print = 0

    def progress_callback(processed_frames: int, total_frames: int) -> None:
        nonlocal last_percent, last_count_print

        if total_frames > 0:
            percent = int((processed_frames * 100) / total_frames)
            if percent >= last_percent + 5 or percent == 100:
                print(
                    f"Processing: {percent}% ({processed_frames}/{total_frames} frames)",
                    end="\r",
                    flush=True,
                )
                last_percent = percent
        else:
            if processed_frames - last_count_print >= 100:
                print(f"Processing: {processed_frames} frames", end="\r", flush=True)
                last_count_print = processed_frames

    return progress_callback


def run_worker_loop() -> None:
    while True:
        # 1. Get task
        response = requests.post(f"{SERVER}/get-task", timeout=30)
        response.raise_for_status()
        data = response.json()

        if "msg" in data:
            print("No tasks available, sleeping 5s...")
            time.sleep(5)
            continue

        task_code = data["task_code"]
        download_url = data["download_url"]

        # 2. Download video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as file_handle:
            tmp_path = file_handle.name

        print(f"Task {task_code}: downloading video")
        download_file_with_progress(download_url, tmp_path, timeout=300)

        try:
            # 3. Process to .npy
            print(f"Task {task_code}: processing video")
            progress_callback = build_process_progress_callback()
            results = process_video(tmp_path, progress_callback=progress_callback)
            print("Processing: 100%", " " * 30)

            # 4. Send back
            buffer = io.BytesIO()
            np.save(buffer, results)
            print(f"Task {task_code}: uploading result")
            send_result_with_retry(SERVER, task_code, buffer)

            print(f"Done: {task_code}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


def check_status() -> None:
    response = requests.get(f"{SERVER}/status", timeout=30)
    response.raise_for_status()
    data = response.json()
    total = data.get('total',0)
    print("Server status:")
    print(f"  total: {data.get('total', 0)}")
    print(f"  processed: {data.get('processed', 0)}")
    print(f"  processing: {data.get('processing', 0)}")
    print(f"  pending: {data.get('pending', 0)}")
    print(f"  % complete: {(data.get('processed',0)/total if total > 0 else 0)*100}%")


def download_annotations() -> None:
    ANNOTATIONS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading annotations ZIP")
    download_file_with_progress(f"{SERVER}/download-annotations", str(ANNOTATIONS_OUTPUT), timeout=600)

    print(f"Saved keypoints ZIP to {ANNOTATIONS_OUTPUT}")


def run_mode(mode: str) -> None:
    if mode == "worker":
        run_worker_loop()
    elif mode == "status":
        check_status()
    elif mode == "download":
        download_annotations()
    else:
        raise ValueError(f"Unknown mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hive worker CLI")
    parser.add_argument(
        "--mode",
        choices=["worker", "status", "download-annotations"],
        required=True,
        help="Run mode",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run_mode(args.mode)
    except requests.RequestException as exc:
        print(f"Network error: {exc}")
    except Exception as exc:
        print(f"Error: {exc}")

if __name__ == "__main__":
    main()

