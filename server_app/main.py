from fastapi import FastAPI
import uvicorn
from database import Database
import os


VIDEO_PATH = "./dataset/videos"


app = FastAPI()

#Get processing status
@app.get("/status")
async def get_status():
    return{
        "health": "ok",
        "processed": 2,
        "total": 500
    }



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
    URL="Swag"

    db = Database(URL)
    db.create_tables()
    find_videos(db)

    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()