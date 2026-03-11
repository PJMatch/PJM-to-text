from datetime import datetime
from sqlalchemy import String,Boolean,ForeignKey,DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import DeclarativeBase,relationship,mapped_column,Mapped
import uuid

class Base(DeclarativeBase):
    pass

#File table
class File(Base):
    __tablename__ = "files"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255),unique=True, nullable=False)
    path: Mapped[str] = mapped_column(String(500), nullable=False)
    is_processed: Mapped[bool] = mapped_column(Boolean, default=False)
    is_processing: Mapped[bool] = mapped_column(Boolean, default=False)

    tasks: Mapped[list["Task"]] = relationship(back_populates="file")

    def __repr__(self):
        return f"File(name={self.name!r}, processed={self.is_processed})"



#Task table
class Task(Base):
    __tablename__ = "tasks"
    id: Mapped[int] = mapped_column(primary_key=True)
    unique_code: Mapped[str] = mapped_column(
        String(50), 
        unique=True, 
        default=lambda: str(uuid.uuid4())[:8]
    )
    file_id: Mapped[int] = mapped_column(ForeignKey("files.id"), nullable=False)

    file: Mapped["File"] = relationship(back_populates="tasks")

    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    def __repr__(self):
        return f"Task(code={self.unique_code!r}, file_id={self.file_id})"