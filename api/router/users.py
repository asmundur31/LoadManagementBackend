'''
    This module is for all endpoints under the /users endpoint.
'''
import os
import shutil
from fastapi import APIRouter, HTTPException, Depends
from typing import List
from sqlalchemy.orm import Session

import api.schemas as schemas
import api.models as models
from api.database import get_db

router = APIRouter(
    prefix="/users",
    tags=["Users"]
)

# Get all users
@router.get('/', response_model=List[schemas.Users])
def get_all_users(db: Session = Depends(get_db)):
    users = db.query(models.User).all()
    return users

# Add a new user
@router.post('/', response_model=schemas.Users)
def add_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    # Check if user_name already exists
    existing_user = db.query(models.User).filter(models.User.user_name == user.user_name).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="User name already exists")

    new_user = models.User(**user.dict())
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

# Edit a user's user_name
@router.put('/{user_id}', response_model=schemas.Users)
def edit_user(user_id: int, user_update: schemas.UserUpdate, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check if the new user_name is already taken
    if db.query(models.User).filter(models.User.user_name == user_update.user_name).first():
        raise HTTPException(status_code=400, detail="User name already exists")

    # Then we save the new name to the database
    user.user_name = user_update.user_name
    db.commit()
    db.refresh(user)
    return user


# Get user's uploaded recordings
@router.get('/{user_id}/recordings', response_model=List[schemas.RecordingWithUser])
def get_recordings(user_id: int, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Fetch all recordings associated with the user
    recordings = (
        db.query(models.Recording)
        .join(models.User)  # Join User to get user_name
        .filter(models.Recording.user_id == user_id)
        .all()
    )
    result = [
        schemas.RecordingWithUser(
            recording_id=recording.id,
            recording_name=recording.recording_name,
            user_name=recording.user.user_name,
            uploaded_at=recording.uploaded_at.isoformat()
        ) 
        for recording in recordings
    ]
    return result

# Delete user's recording
@router.delete('/{user_id}/recordings/{recording_id}', response_model=schemas.RecordingWithUser)
def delete_recordings(user_id: int, recording_id: int, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    recording = db.query(models.Recording).filter(models.Recording.user_id == user_id, models.Recording.id == recording_id).first()
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")
    
    # Store the recording details before deleting
    deleted_recording = schemas.RecordingWithUser(
        recording_id=recording.id,
        recording_name=recording.recording_name,
        user_name=recording.user.user_name,
        uploaded_at=recording.uploaded_at.isoformat()
    )

    # Construct the directory path
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw'))
    dir_path = os.path.join(base_dir, str(user_id), recording.recording_name)
    
    # Delete the recording directory from the filesystem
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    
    # Delete the recording
    db.delete(recording)
    db.commit()
    
    return deleted_recording
