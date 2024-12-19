'''
    This module is to define the database tables.
'''
from sqlalchemy import Column, Integer, String, TIMESTAMP, Boolean, text, ForeignKey
from sqlalchemy.orm import relationship

from api.database import Base


class Post(Base):
    __tablename__ = "posts"

    id = Column(Integer,primary_key=True,nullable=False)
    title = Column(String,nullable=False)
    content = Column(String,nullable=False)
    published = Column(Boolean, server_default='TRUE')
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'))


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    user_name = Column(String, nullable=False)

    recordings = relationship("Recording", back_populates="user", cascade="all, delete-orphan")


class Recording(Base):
    __tablename__ = "recordings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    recording_name = Column(String, nullable=False)
    uploaded_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'))

    user = relationship("User", back_populates="recordings")
    uploads = relationship("Upload", back_populates="recording", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "recording_name": self.recording_name,
            "uploaded_at": self.uploaded_at.isoformat() if self.uploaded_at else None
        }
    
    
class Upload(Base):
    __tablename__ = "uploads"

    id = Column(Integer, primary_key=True, index=True)
    recording_id = Column(Integer, ForeignKey("recordings.id", ondelete="CASCADE"), nullable=False)
    filename = Column(String, nullable=False)
    path = Column(String, nullable=False)

    recording = relationship("Recording", back_populates="uploads")