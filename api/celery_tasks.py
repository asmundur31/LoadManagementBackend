from celery import shared_task
import zipfile
import pathlib
import shutil
import os
import json

from api.calc.fib import fib
from api.models import Upload, Recording
from api.database import get_db_session
from api.utils import convert_annotation_file_to_json, convert_sensor_directory_to_json, fix_timestamps


@shared_task
def upload_recording_to_db(user_id: str, recording_name: str) -> str:
    """
        Uploads the files in the extracted directory.
    """
    print("Uploading recording to db...")
    try:
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

        return new_recording.to_dict()
    except Exception as e:
        raise RuntimeError(f"Failed to upload data: {e}")


@shared_task
def extract_zip_file(recording: dict, zip_path: str, user_dir: str) -> tuple:
    """
    Extracts a zip file, converts CSV files to JSON, and removes the original CSV files.
    """
    try:
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(user_dir)
        
        # Clean up temporary zip file
        pathlib.Path(zip_path).unlink(missing_ok=True)

        # Remove __MACOSX folder if present
        macosx_folder = pathlib.Path(user_dir) / "__MACOSX"
        if macosx_folder.exists():
            shutil.rmtree(macosx_folder)

        # Convert CSV files to JSON
        target_path = os.path.join(user_dir, recording["recording_name"])
        
        # Process the annotation file
        date_dir = next(pathlib.Path(target_path).iterdir())
        annotation_files = list(date_dir.glob("Annotations*.csv"))
        if annotation_files:
            convert_annotation_file_to_json(annotation_files[0], recording)
        
        # Process each sensor directory
        for sensor_dir in date_dir.iterdir():
            if sensor_dir.is_dir():
                convert_sensor_directory_to_json(sensor_dir, recording)

        return {
            "target_path": target_path,
            "recording": recording
        }
    except Exception as e:
        raise RuntimeError(f"Failed to extract and process zip file: {e}")

@shared_task
def upload_files_to_db(path_recording: dict) -> dict:
    """
        Uploads the files in the extracted directory.
    """
    print("Uploading files to db...")
    target_path = path_recording["target_path"]
    recording = path_recording["recording"]
    try:
        processed_files = []
        uploads = []

        with get_db_session() as db:
            for file_path in pathlib.Path(target_path).rglob("*"):
                if file_path.is_file() and file_path.suffix in {".json", ".mov"}:
                    new_upload = Upload(
                        recording_id=recording["id"],
                        filename=file_path.name,
                        path=str(file_path)
                    )
                    uploads.append(new_upload)
                    processed_files.append(file_path.name)
            db.add_all(uploads)
            db.commit()

        return path_recording 
    except Exception as e:
        raise RuntimeError(f"Failed to upload data: {e}")

@shared_task
def process_data(path_recording: dict) -> str:
    """
        Processes the data of a recording
    """
    recording = path_recording["recording"]
    target_path = path_recording["target_path"]
    print(f"Processing data of recording with id={recording['id']}...")

    # Fetch and process each JSON file
    for json_file in pathlib.Path(target_path).rglob("*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        if "data" in data:
            sensor_data = data["data"]
            timestamps = sensor_data["timestamp"]

            # Implement your logic to fix the timestamps here
            fixed_timestamps = fix_timestamps(timestamps)
            data["data"]["timestamp"] = fixed_timestamps

            # Save the updated JSON data back to the file
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=4)

    return f"Finished processing recording {recording['id']}"

@shared_task
def dummy_test(num: int):
    print("Starting a dummy process...")
    # Simulate long task
    result = fib(num)
    return f"The {num}th fibonacci number is {result}"
