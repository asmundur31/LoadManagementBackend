'''
    This module is for all endpoints under the /recordings endpoint.
'''
import os
import zipfile
import shutil
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from typing import List
from sqlalchemy.orm import Session

import api.schemas as schemas
import api.models as models
from api.database import get_db

router = APIRouter(
    prefix="/recordings",
    tags=["Recordings"]
)

# Get all recordings
@router.get('/', response_model=List[schemas.Recording])
def get_all_recordings(db: Session = Depends(get_db)):
    recordings = db.query(models.Recording).all()
    return recordings


# Get recording by id 
@router.get('/{recording_id}', response_model=schemas.RecordingWithZip)
def get_recording(recording_id: int, db: Session = Depends(get_db)):
    recording = db.query(models.Recording).filter(models.Recording.id == recording_id).first()
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")
    
    # Construct the directory path
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw'))
    dir_path = os.path.join(base_dir, str(recording.user_id), recording.recording_name)
    
    if not os.path.exists(dir_path):
        raise HTTPException(status_code=404, detail="Recording directory not found")
    
    # Create a zip file
    zip_filename = f"{recording.recording_name}_{recording.id}.zip"
    zip_filepath = os.path.join(base_dir, zip_filename)
    with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=dir_path)
                zipf.write(file_path, arcname=arcname)
    
    # Construct the zip file URL
    zip_url = f"/recordings/download/{zip_filename}"
    
    # Return the recording data and the zip file URL
    return {
        "recording": recording,
        "zip_url": zip_url
    }

# Endpoint to download the zip file
@router.get('/download/{zip_filename}', response_class=FileResponse)
def download_zip(zip_filename: str):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw'))
    zip_filepath = os.path.join(base_dir, zip_filename)
    if not os.path.exists(zip_filepath):
        raise HTTPException(status_code=404, detail="Zip file not found")
    return FileResponse(zip_filepath, filename=zip_filename, media_type='application/zip')

# Delete recording by id
@router.delete('/{recording_id}', response_model=schemas.Recording)
def delete_recording(recording_id: int, db: Session = Depends(get_db)):
    recording = db.query(models.Recording).filter(models.Recording.id == recording_id).first()
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")
    
    # Construct the directory path
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw'))
    dir_path = os.path.join(base_dir, str(recording.user_id), recording.recording_name)
    
    # Delete the directory and its contents
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    
    # Delete the recording from the database
    db.delete(recording)
    db.commit()
    
    return recording