'''
    This module is to define the schema for data validation.
'''
from pydantic import BaseModel
from datetime import datetime

### User schemas
class Users(BaseModel):
    id: int
    user_name: str

    class Config:
        from_attributes = True

class UserCreate(BaseModel):
    user_name: str

    class Config:
        from_attributes = True

class UserUpdate(BaseModel):
    user_name: str

    class Config:
        from_attributes = True

### Recording schemas
class Recording(BaseModel):
    id: int
    user_id: int
    recording_name: str
    uploaded_at: datetime 

    class Config:
        from_attributes = True

class RecordingWithZip(BaseModel):
    recording: Recording
    zip_url: str

    class Config:
        from_attributes = True


class RecordingWithUser(BaseModel):
    recording_id: int
    recording_name: str
    user_name: str
    uploaded_at: str

    class Config:
        from_attributes = True

class RecordingWithZipAndUser(BaseModel):
    recording: RecordingWithUser
    zip_url: str

    class Config:
        from_attributes = True

class FileMetadata(BaseModel):
    filename: str
    path: str

    class Config:
        from_attributes = True


class UploadResponse(BaseModel):
    message: str
    user_name: str
    recording_name: str
    task_id: str

    class Config:
        from_attributes = True
