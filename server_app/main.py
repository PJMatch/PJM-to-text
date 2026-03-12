import os
import shutil
import zipfile

import uvicorn
from fastapi import Depends, FastAPI, File as FastAPIFile, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from database import Database
from models import File, Task


VIDEO_PATH = os.getenv("VIDEOS_DIR", "/data/videos")
EXTRACTED_PATH = os.getenv("EXTRACTED_DIR", "/data/extracted")
ZIP_OUTPUT_PATH = os.getenv("ZIP_OUTPUT_DIR", "/data/processing_result")
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")
ANNOTATIONS_ZIP_NAME = os.getenv("ANNOTATIONS_ZIP_NAME", "annotations_final.zip")


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
    async def finish_task(
        task_code: str,
        file: UploadFile = FastAPIFile(...),
        session: Session = Depends(db.get_db),
    ):
        task, state = db.mark_task_as_completed(session, task_code)
        if state == "not_found":
            raise HTTPException(status_code=404, detail="error: task not found")
        if state in {"not_in_progress", "invalid_file_state"}:
            raise HTTPException(status_code=409, detail="error: task is not active")

        try:
            video_file = task.file
            extracted_filename = f"{video_file.name}.npy"
            extracted_path = os.path.join(EXTRACTED_PATH, extracted_filename)

            os.makedirs(EXTRACTED_PATH, exist_ok=True)
            with open(extracted_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            session.commit()

            return {
                "status": "success",
                "message": f"Task {task_code} finished, result saved as {extracted_filename}",
                "total_processed": session.query(File).filter(File.is_processed == True).count()
            }
        except Exception as exc:
            session.rollback()
            raise HTTPException(status_code=500, detail=f"error: failed to store result ({exc})")
    ###############################################################

    @app.get("/download-annotations")
    async def download_annotations(session: Session = Depends(db.get_db)):
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
            raise HTTPException(status_code=404, detail="error: no annotation files found")

        zip_path = os.path.join(ZIP_OUTPUT_PATH, ANNOTATIONS_ZIP_NAME)

        if not os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
                for filename in sorted(npy_files):
                    file_path = os.path.join(EXTRACTED_PATH, filename)
                    zipf.write(file_path, arcname=filename)

        return FileResponse(
            path=zip_path,
            media_type="application/zip",
            filename=ANNOTATIONS_ZIP_NAME,
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