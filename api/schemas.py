'''
    This module is to define the schema for data validation.
'''
from pydantic import BaseModel
from typing import List


class PostBase(BaseModel):
    content: str
    title: str

    class Config:
        orm_mode = True


class CreatePost(PostBase):
    class Config:
        orm_mode = True


class FileMetadata(BaseModel):
    filename: str
    path: str

    class Config:
        orm_mode = True


class UploadResponse(BaseModel):
    message: str
    user_name: str
    recording_name: str
    files: List[FileMetadata]

    class Config:
        orm_mode = True


class Users(BaseModel):
    id: int
    user_name: str

    class Config:
        orm_mode = True

class UserCreate(BaseModel):
    user_name: str

    class Config:
        orm_mode = True

class UserUpdate(BaseModel):
    user_name: str

    class Config:
        orm_mode = True