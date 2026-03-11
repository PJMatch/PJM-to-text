from fastapi import FastAPI,Depends, HTTPException, UploadFile, File as FastAPIFile
import uvicorn
from database import Database
from models import File, Task
from sqlalchemy.orm import Session
import os
from fastapi.responses import FileResponse
import shutil


VIDEO_PATH = os.getenv("VIDEOS_DIR", "/data/videos")
EXTRACTED_PATH = os.getenv("EXTRACTED_DIR", "/data/extracted")

def create_app(db_url: str):

    app = FastAPI()
    db = Database(db_url)

    @app.get("/status")
    async def get_status(session: Session = Depends(db.get_db)):
        return {"total": session.query(File).count()}

    @app.post("/get-task")
    async def get_task(session: Session = Depends(db.get_db)):
        task = db.add_new_task(session)
        if not task:
            return {"msg": "no0 tasks"}
    
        session.commit()

        return {
            "task_code": task.unique_code,
            "video_name": task.file.name,
            "download_url": f"http://localhost:8000/download-task/{task.unique_code}"
        }

    @app.get("/download-task/{task_code}")
    async def download_task_file(task_code: str,session: Session = Depends(db.get_db)):
        task = session.query(Task).filter(Task.unique_code == task_code).first()

        if not task:
            raise HTTPException(status_code=404, detail="error: not found")
        
        file_path = task.file.path

        if not os.path.exists(file_path):
            raise HTTPException(status_code=503, detail="error: server error")
                                
        return FileResponse(
        path=file_path, 
        media_type="video/mp4", 
        filename=task.file.name
    )

    @app.post("/finish-task/{task_code}")
    async def finish_task(
        task_code: str, 
        file: UploadFile = FastAPIFile(...),
        session: Session = Depends(db.get_db)
        ):

        task = db.mark_task_as_completed(session, task_code)
        if not task:
            raise HTTPException(status_code=404, detail="error: task not found")
        
        video_file = task.file

        extracted_filename = f"{video_file.name}.npy"

        extracted_path = os.path.join(EXTRACTED_PATH, extracted_filename)

        with open(extracted_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)


        session.commit()

        return {
            "status": "success",
            "message": f"Task {task_code} finished, result saved as {extracted_filename}",
            "total_processed": session.query(File).filter(File.is_processed == True).count()
            }

    return app, db


def find_videos(db: Database):

    if not os.path.exists(VIDEO_PATH):
        print("error")
        return

    videos = [f for f in os.listdir(VIDEO_PATH) if f.endswith('.mp4')]

    session = db.get_db_session()

    try: 
        for vidname in videos:

            if not db.get_file_by_name(session,vidname):
                full_path = os.path.join(VIDEO_PATH, vidname)
                db.add_new_file(session,vidname,full_path)
        
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