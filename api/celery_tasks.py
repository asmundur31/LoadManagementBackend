from celery import shared_task
import zipfile
import pathlib
import shutil

from api.calc.fib import fib
from api.models import Upload, Recording
from api.database import get_db_session


@shared_task
def extract_zip_file(zip_path: str, extract_to: str) -> str:
    """
    Extracts a zip file to the specified directory.
    """
    print('Start extraction')
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("extracted the zip")
        # Clean up temporary zip file
        pathlib.Path(zip_path).unlink(missing_ok=True)

        # Remove __MACOSX folder if present
        macosx_folder = pathlib.Path(extract_to) / "__MACOSX"
        if macosx_folder.exists():
            shutil.rmtree(macosx_folder)

        return extract_to  # Pass extracted directory to the next task
    except Exception as e:
        raise RuntimeError(f"Failed to extract zip file: {e}")

@shared_task
def upload_to_db(extracted_dir: str, user_id: str, recording_name: str) -> str:
    """
        Uploads the files in the extracted directory.
    """
    print("Uploading data to db...")
    recording_dir = pathlib.Path(extracted_dir) / recording_name 
    try:
        processed_files = []
        uploads = []

        # Use the database session
        with get_db_session() as db:
            new_recording = Recording(
                user_id=user_id,
                recording_name=recording_name
            )
            db.add(new_recording)
            db.commit()
            # Then we need to generate the id for the new recording
            db.refresh(new_recording)
        with get_db_session() as db:
            for file_path in recording_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix in {".csv", ".mov"}:
                    new_upload = Upload(
                        recording_id=new_recording.id,
                        filename=file_path.name,
                        path=str(file_path)
                    )
                    uploads.append(new_upload)
                    processed_files.append(file_path.name)
            db.add_all(uploads)
            db.commit()

        return new_recording.id
    except Exception as e:
        raise RuntimeError(f"Failed to upload data: {e}")

@shared_task
def process_data(recording_id: str):
    """
        Processes the data of a recording
    """
    print(f"Processing data of recording with id={recording_id}...")
    return f"Finnished processing recording {recording_id}"

@shared_task
def dummy_test(num: int):
    print("Starting a dummy process...")
    # Simulate long task
    result = fib(num)
    return f"The {num}th fibonacci number is {result}"
