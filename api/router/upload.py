'''
    This module is for all endpoints under the /upload endpoint.
'''
import os
import io
import shutil
import zipfile
import pathlib
from fastapi import APIRouter, UploadFile, HTTPException, Depends, Form
from sqlalchemy.orm import Session

from api.database import get_db
from api.models import Upload, User
from api.schemas import UploadResponse

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
    Endpoint to upload a zipped directory.
    - The zipped file will be extracted into `data/raw/`.
    - Metadata for each extracted file will be stored in the database.
    """
   
    # Check if uploaded file is a zip
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only .zip files are allowed")

    # Get the user
    user = db.query(User).filter(User.id == user_id).one()
    if not user:
        raise HTTPException(status_code=404, detail=f"User not found. No user with user_id={user_id}.")

    USER_DIR = UPLOAD_DIR+f'{user.id}/'
    if not os.path.exists(USER_DIR):
        os.makedirs(USER_DIR)

    try:
        # Create a sub-directory for each zip upload to extract the files into
        zip_upload_dir = pathlib.Path(USER_DIR) / pathlib.Path(file.filename).stem
        zip_upload_dir.mkdir(parents=True, exist_ok=True)
        # Read the uploaded zip file into memory
        zip_data = await file.read()
        with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zip_ref:
            # Extract all files to the specified directory
            zip_ref.extractall(USER_DIR)

            # Remove __MACOSX folder if present
            macosx_folder = pathlib.Path(USER_DIR) / pathlib.Path("__MACOSX")
            if macosx_folder.exists() and macosx_folder.is_dir():
                shutil.rmtree(macosx_folder)

            # Save metadata for each extracted file to the database
            extracted_files = []
            for filename in zip_ref.namelist():
                # Ignore __MACOSX
                if filename.startswith('__MACOSX'):
                    continue
                elif filename.endswith(('.csv', '.mov')):
                    # Only process files extracted from the zip, not the whole UPLOAD_DIR
                    file_path = pathlib.Path(USER_DIR) / filename
                    # Save the metadata to the database
                    new_upload = Upload(
                        user_id=user_id,
                        recording_name=recording_name,
                        filename=os.path.basename(filename),
                        path=str(file_path)
                    )
                    db.add(new_upload)
                    db.commit()
                    extracted_files.append({"filename": filename, "path": str(file_path)})


        # Return a success response
        return UploadResponse(
            message=f"Files extracted and saved successfully",
            user_name=user.user_name,
            recording_name=recording_name,
            files=extracted_files
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the file: {str(e)}")
