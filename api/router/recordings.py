'''
    This module is for all endpoints under the /recordings endpoint.
'''
from fastapi import APIRouter, Depends
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
@router.get('/{recording_id}', response_model=schemas.Recording)
def get_all_recordings(recording_id: int, db: Session = Depends(get_db)):
    recording = db.query(models.Recording).filter(models.Recording.id == recording_id).first()
    return recording

