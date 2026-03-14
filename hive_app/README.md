# Hiveprocess App

Worker pulls video tasks from API server, processes them, and uploads `.npy` results.

## 1. Install dependencies

```bash
pip install -r hive_app/requirements.txt
```

Installs worker requirements (`requests`, `numpy`, `mediapipe`, `opencv-python`).

## 2. Go to workdir

```bash
cd hive_app
```

## 3. Run worker loop

```bash
python main.py --mode worker --pass PASSWORD
```

## 4. Run multiple workers

Start workers in multiple terminals:

```bash
python main.py --mode worker --pass PASSWORD
python main.py --mode worker --pass PASSWORD
python main.py --mode worker --pass PASSWORD
```

Each process pulls different tasks and increases total throughput.
Recommended: about 1 worker per 2 CPU cores.

## 5. Check server status

```bash
python main.py --mode status --pass PASSWORD
```

Prints `total`, `processed`, `processing`, `pending`, and completion percent.

## 6. Download keypoints ZIP

```bash
python main.py --mode download --pass PASSWORD
```

Downloads ZIP from server to local `keypoints.zip`.
Works only after processing is finished.

## Notes

- Worker uses HTTPS endpoint configured in code (`https://hiveprocess.duckdns.org`).
- For long uploads, set `UPLOAD_TIMEOUT_SECONDS` (default: `7200`).
