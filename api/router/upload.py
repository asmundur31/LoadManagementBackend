'''
    This module is for all endpoints under the /upload endpoint.
'''
import pathlib
from fastapi import APIRouter, UploadFile, HTTPException, Depends, Form
from sqlalchemy.orm import Session

from api.database import get_db
from api.models import Recording, User
from api.schemas import UploadResponse
import api.celery_tasks as ct

router = APIRouter(
    prefix="/upload",
    tags=["Upload"]
)

UPLOAD_DIR = "data/raw/"

@router.post("/{user_id}", response_model=UploadResponse)
async def upload_directory(
    file: UploadFile,
    user_id: int,
    recording_name: str = Form(...),
    db: Session = Depends(get_db)
):
    """
        Endpoint to upload a zipped directory and trigger Celery tasks.
    """
    # Validate file type
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only .zip files are allowed")

    # Ensure user exists
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"User with ID {user_id} not found.")

    # Ensure user does not have recording name that exist
    recording = db.query(Recording).filter(Recording.user_id == user.id, Recording.recording_name == recording_name).first()
    if recording:
        raise HTTPException(status_code=404, detail=f"Recording with recording name {recording_name} already exists.")
    
    try:
        # Define user directory
        user_dir = pathlib.Path(UPLOAD_DIR) / str(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        # Save zip file temporarily for extraction
        temp_zip_path = user_dir / file.filename
        with open(temp_zip_path, "wb") as temp_zip_file:
            temp_zip_file.write(await file.read())

        # Trigger Celery chain
        chain_result = (
            ct.upload_recording_to_db.s(user_id=user_id, recording_name=recording_name) |
            ct.extract_zip_file.s(str(temp_zip_path), str(user_dir)) |
            ct.upload_files_to_db.s() |
            ct.process_data.s()
        ).apply_async()

        # Return success response
        return UploadResponse(
            message="Data uploaded successfully and processing has started.",
            user_name=user.user_name,
            recording_name=recording_name,
            task_id=chain_result.id,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the upload: {str(e)}")
