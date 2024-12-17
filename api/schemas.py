'''
    This module is to define the schema for data validation.
'''
from pydantic import BaseModel
from typing import List
from datetime import date

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
    recording_id: int
    recording_name: str
    user_name: str
    uploaded_at: str

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
