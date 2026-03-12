import io
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import requests

from mediapipe_process import process_video

SERVER = os.getenv("SERVER_URL", "http://localhost:8000")
ANNOTATIONS_OUTPUT = Path(os.getenv("ANNOTATIONS_OUTPUT", "annotations.zip"))




def send_result_with_retry(server: str, task_code: str, buf: io.BytesIO, retries: int = 3) -> None:
    url = f"{server}/finish-task/{task_code}"
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            buf.seek(0)
            response = requests.post(
                url,
                files={"file": ("result.npy", buf, "application/octet-stream")},
                timeout=120,
            )
            response.raise_for_status()
            return
        except requests.RequestException as exc:
            last_error = exc
            print(f"error upload fail {attempt}/{retries} for {task_code}: {exc}")
            if attempt < retries:
                time.sleep(2 * attempt)

    raise RuntimeError(f"error upload failed - task {task_code}: {last_error}")


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
        download_response = requests.get(download_url, timeout=300)
        download_response.raise_for_status()
        video_bytes = download_response.content

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as file_handle:
            file_handle.write(video_bytes)
            tmp_path = file_handle.name

        try:
            # 3. Process to .npy
            results = process_video(tmp_path)

            # 4. Send back
            buffer = io.BytesIO()
            np.save(buffer, results)
            send_result_with_retry(SERVER, task_code, buffer)

            print(f"Done: {task_code}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


def check_status() -> None:
    response = requests.get(f"{SERVER}/status", timeout=30)
    response.raise_for_status()
    data = response.json()
    print("Server status:")
    print(f"  total: {data.get('total', 0)}")
    print(f"  processed: {data.get('processed', 0)}")
    print(f"  processing: {data.get('processing', 0)}")
    print(f"  pending: {data.get('pending', 0)}")


def download_annotations() -> None:
    response = requests.get(f"{SERVER}/download-annotations", timeout=600)
    response.raise_for_status()

    ANNOTATIONS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(ANNOTATIONS_OUTPUT, "wb") as out_file:
        out_file.write(response.content)

    print(f"Saved annotations ZIP to {ANNOTATIONS_OUTPUT}")


def menu() -> None:
    while True:
        print("\nChoose an option:")
        print("1) Run worker loop")
        print("2) Check server status")
        print("3) Download annotations ZIP")
        print("4) Exit")
        choice = input("> ").strip()

        try:
            if choice == "1":
                run_worker_loop()
            elif choice == "2":
                check_status()
            elif choice == "3":
                download_annotations()
            elif choice == "4":
                print("Bye")
                return
            else:
                print("Unknown option, choose 1-4")
        except requests.RequestException as exc:
            print(f"Network error: {exc}")
        except Exception as exc:
            print(f"Error: {exc}")


def main() -> None:
    menu()

if __name__ == "__main__":
    main()

