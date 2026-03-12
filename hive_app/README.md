# Hiveprocess App

Worker pulls video tasks from API server, processes them, and uploads `.npy` results.

## 1. Install dependencies

```bash
pip install -r hive_app/requirements.txt
```

Installs worker requirements (`requests`, `numpy`, `mediapipe`, `opencv-python`).


## 1,5. Goto hive_app directory

```bash
cd hive_app
```

## 2. Run worker loop

```bash
python hive_app/main.py --mode worker
```

Continuously fetches tasks, downloads video, processes frames, uploads `.npy`.

## 3. Run multiple worker processes

Start worker in multiple terminals:

```bash
python hive_app/main.py --mode worker
python hive_app/main.py --mode worker
python hive_app/main.py --mode worker
```

Each process pulls different tasks and increases total throughput. Recommended 1 worker/ 2 CPU cores

## ad 1. Check server status

```bash
python hive_app/main.py --mode status
```

Prints `total`, `processed`, `processing`, `pending`, and completion percent.

## ad 2. Download keypoints ZIP

```bash
python hive_app/main.py --mode download
```

Downloads ZIP from server to local `keypoints.zip`. Can be done only after extraction is finished.

