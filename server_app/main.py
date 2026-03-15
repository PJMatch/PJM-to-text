import os
import shutil
import zipfile

import uvicorn
from fastapi import Depends, FastAPI, File as FastAPIFile, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from database import Database
from models import File, Task, TaskStatus


VIDEO_PATH = os.getenv("VIDEOS_DIR", "/data/videos")
EXTRACTED_PATH = os.getenv("EXTRACTED_DIR", "/data/extracted")
ZIP_OUTPUT_PATH = os.getenv("ZIP_OUTPUT_DIR", "/data/processing_result")
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")
KEYPOINTS_ZIP_NAME = os.getenv(
    "KEYPOINTS_ZIP_NAME",
    os.getenv("ANNOTATIONS_ZIP_NAME", "keypoints_final.zip"),
)


def create_app(db_url: str):

    
    app = FastAPI()
    db = Database(db_url)

    @app.get("/status")
    async def get_status(session: Session = Depends(db.get_db)):
        total = session.query(File).count()
        processed = session.query(File).filter(File.is_processed == True).count()
        processing = session.query(File).filter(
            File.is_processing == True,
            File.is_processed == False,
        ).count()
        pending = total - processed - processing

        return {
            "total": total,
            "processed": processed,
            "processing": processing,
            "pending": pending,
        }

    @app.get("/tasks/in-progress")
    async def get_in_progress_tasks(session: Session = Depends(db.get_db)):
        tasks = session.query(Task).filter(Task.status == TaskStatus.IN_PROGRESS).order_by(Task.id).all()
        return {
            "count": len(tasks),
            "task_ids": [task.id for task in tasks],
            "tasks": [
                {
                    "id": task.id,
                    "task_code": task.unique_code,
                    "file_id": task.file_id,
                    "file_name": task.file.name,
                    "created_at": task.created_at.isoformat() if task.created_at else None,
                }
                for task in tasks
            ],
        }

    @app.get("/files")
    async def list_files(session: Session = Depends(db.get_db)):
        files = session.query(File).order_by(File.id).all()
        return {
            "count": len(files),
            "files": [
                {
                    "id": file.id,
                    "name": file.name,
                    "is_processed": file.is_processed,
                    "is_processing": file.is_processing,
                    "has_keypoints": os.path.exists(os.path.join(EXTRACTED_PATH, f"{file.name}.npy")),
                }
                for file in files
            ],
        }

    @app.get("/files/{file_id}/download-keypoints")
    async def download_keypoints_file(file_id: int, session: Session = Depends(db.get_db)):
        file_record = session.query(File).filter(File.id == file_id).first()

        if not file_record:
            raise HTTPException(status_code=404, detail="error: file not found")

        keypoints_path = os.path.join(EXTRACTED_PATH, f"{file_record.name}.npy")
        if not os.path.exists(keypoints_path):
            raise HTTPException(status_code=404, detail="error: keypoints file not found")

        return FileResponse(
            path=keypoints_path,
            media_type="application/octet-stream",
            filename=f"{file_record.name}.npy",
        )
    ####################################################

    @app.post("/get-task")
    async def get_task(session: Session = Depends(db.get_db)):
        task = db.add_new_task(session)
        if not task:
            return {"msg": "no0 tasks"}

        session.commit()

        return {
            "task_code": task.unique_code,
            "video_name": task.file.name,
            "download_url": f"{SERVER_URL}/download-task/{task.unique_code}"
        }
    ############################################

    @app.get("/download-task/{task_code}")
    async def download_task_file(task_code: str, session: Session = Depends(db.get_db)):
        task = session.query(Task).filter(Task.unique_code == task_code).first()

        if not task:
            raise HTTPException(status_code=404, detail="error: not found")
        
        file_path = task.file.path

        if not os.path.exists(file_path):
            raise HTTPException(status_code=503, detail="error: server error")

        return FileResponse(
            path=file_path,
            media_type="video/mp4",
            filename=task.file.name,
        )
    ######################################################

    @app.post("/finish-task/{task_code}")
    def finish_task(
        task_code: str,
        file: UploadFile = FastAPIFile(...),
    ):
        precheck_session = db.get_db_session()
        state = "not_found"
        extracted_filename = ""

        try:
            task, state = db.get_task_for_completion(precheck_session, task_code)
            if state == "ok":
                extracted_filename = f"{task.file.name}.npy"
        finally:
            precheck_session.close()

        if state == "not_found":
            raise HTTPException(status_code=404, detail="error: task not found")
        if state in {"not_in_progress", "invalid_file_state"}:
            raise HTTPException(status_code=409, detail="error: task is not active")

        extracted_path = os.path.join(EXTRACTED_PATH, extracted_filename)
        tmp_path = f"{extracted_path}.part"

        try:
            os.makedirs(EXTRACTED_PATH, exist_ok=True)
            with open(tmp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            os.replace(tmp_path, extracted_path)

            complete_session = db.get_db_session()
            try:
                completed_task, completed_state = db.mark_task_as_completed(complete_session, task_code)
                if completed_state == "not_found":
                    raise HTTPException(status_code=404, detail="error: task not found")
                if completed_state in {"not_in_progress", "invalid_file_state"}:
                    raise HTTPException(status_code=409, detail="error: task is not active")

                complete_session.commit()

                return {
                    "status": "success",
                    "message": f"Task {task_code} finished, result saved as {extracted_filename}",
                    "total_processed": complete_session.query(File).filter(File.is_processed == True).count(),
                }
            except Exception:
                complete_session.rollback()
                raise
            finally:
                complete_session.close()
        except HTTPException:
            raise
        except Exception as exc:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise HTTPException(status_code=500, detail=f"error: failed to store result ({exc})")
        finally:
            file.file.close()

    @app.post("/cancel-task/{task_id}")
    async def cancel_task(task_id: int, session: Session = Depends(db.get_db)):
        task, state = db.cancel_task(session, task_id)
        if state == "not_found":
            raise HTTPException(status_code=404, detail="error: task not found")
        if state == "not_in_progress":
            raise HTTPException(status_code=409, detail="error: task is not active")

        session.commit()
        return {
            "status": "canceled",
            "task_id": task.id,
            "task_code": task.unique_code,
            "file_id": task.file_id,
            "file_name": task.file.name,
        }
    ###############################################################

    @app.get("/download-keypoints")
    async def download_keypoints(session: Session = Depends(db.get_db)):
        total = session.query(File).count()
        processed = session.query(File).filter(File.is_processed == True).count()

        if total == 0:
            raise HTTPException(status_code=404, detail="error: no files in database")

        if processed < total:
            raise HTTPException(
                status_code=409,
                detail=f"error: processing not finished ({processed}/{total})",
            )

        if not os.path.isdir(EXTRACTED_PATH):
            raise HTTPException(status_code=404, detail="error: extracted directory not found")

        npy_files = [f for f in os.listdir(EXTRACTED_PATH) if f.endswith(".npy")]
        if not npy_files:
            raise HTTPException(status_code=404, detail="error: no keypoints files found")

        zip_path = os.path.join(ZIP_OUTPUT_PATH, KEYPOINTS_ZIP_NAME)

        if not os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
                for filename in sorted(npy_files):
                    file_path = os.path.join(EXTRACTED_PATH, filename)
                    zipf.write(file_path, arcname=filename)

        return FileResponse(
            path=zip_path,
            media_type="application/zip",
            filename=KEYPOINTS_ZIP_NAME,
        )

    return app, db


def find_videos(db: Database):
    if not os.path.exists(VIDEO_PATH):
        print("error")
        return

    videos = [f for f in os.listdir(VIDEO_PATH) if f.endswith(".mp4")]
    session = db.get_db_session()

    try:
        for vidname in videos:
            if not db.get_file_by_name(session, vidname):
                full_path = os.path.join(VIDEO_PATH, vidname)
                db.add_new_file(session, vidname, full_path)

        session.commit()
    except Exception as e:
        session.rollback()
        print(f"Error: {e}")
    finally:
        session.close()


def main():
    import time

    URL = os.getenv("DATABASE_URL", "postgresql://hivelord:hivehivehive@db:5432/hive_db")

    for attempt in range(10):
        try:
            app, db = create_app(URL)
            break
        except Exception as e:
            print(f"DB not ready ({attempt+1}/10), retrying... {e}")
            time.sleep(3)
    else:
        raise RuntimeError("error - failed to connect w db")

    db.create_tables()
    find_videos(db)
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()