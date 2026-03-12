import tempfile
import requests
import time
import numpy as np
import os
import io
from mediapipe_process import process_video

SERVER = "http://localhost:8000"



while True:
    #1 get task
    r = requests.post(f"{SERVER}/get-task")
    data = r.json()

    if "msg" in data:
        print("error, 0 tasks left, quitting...")
        time.sleep(5)
        break

    task_code = data["task_code"]
    download_url = data["download_url"]

    #2 download video

    video_bytes = requests.get(download_url).content
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        tmp_path = f.name
    
    try:

        #3. process to .npy

        results = process_video(tmp_path)

        #4. send back
        buf = io.BytesIO()
        np.save(buf, results)
        buf.seek(0)



        requests.post(
            f"{SERVER}/finish-task/{task_code}",
            files={"file": ("result.npy", buf, "application/octet-stream")}
        )
        print(f"Done: {task_code}")
        
    finally:
        os.remove(tmp_path)



