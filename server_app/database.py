from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base, File, Task, TaskStatus
from sqlalchemy.orm import Session
from datetime import datetime, timezone, timedelta

TASK_TIMEOUT_MINUTES = 180


class Database:
    def __init__(self,url):
        try: 
            self.engine = create_engine(url)

            with self.engine.connect() as connection:
                print("db connected")


            self.Session = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False
            )
        except Exception as e:
            print(f"Error {e}")
            raise e
    
    def create_tables(self):
        Base.metadata.create_all(bind=self.engine)

    def get_db_session(self):
        return self.Session()
    
    def get_db(self):
        db = self.Session()
        try:
            yield db
        finally:
            db.close()

    def get_file_by_name(self, session: Session, name: str):
        """checks if file exists in db"""
        return session.query(File).filter(File.name == name).first()
    

    def add_new_file(self, session: Session, name: str, path: str):
        """creates file in db"""
        new_file = File(name=name, path=path)
        session.add(new_file)
        return new_file

    def add_new_task(self,session:Session):

        timeout_threshold = datetime.now(timezone.utc) - timedelta(minutes=TASK_TIMEOUT_MINUTES)
    
        stuck_tasks = session.query(Task).join(File).filter(
            File.is_processing == True,
            File.is_processed == False,
            Task.created_at < timeout_threshold
            ).all()

        for stuck in stuck_tasks:
            stuck.file.is_processing = False
            stuck.status = TaskStatus.FAILED
            

        file_to_process = session.query(File).filter(
            File.is_processed == False,
            File.is_processing == False
        ).with_for_update().first()

        if not file_to_process:
            return None
        
        file_to_process.is_processing = True

        new_task = Task(file_id=file_to_process.id)

        session.add(new_task)

        session.flush()

        return new_task
    
    def mark_task_as_completed(self, session:Session, code: str):

        task = session.query(Task).filter(Task.unique_code == code).first()
        if not task:
            return None
        
        task.status= TaskStatus.SUCCESS
        task.file.is_processed = True
        task.file.is_processing = False
        return task

