from celery import shared_task
import zipfile
import pathlib
import shutil
import os
import json
import pandas as pd
from functools import reduce

from api.calc.fib import fib
from api.models import Upload, Recording
from api.database import get_db_session
from api.utils import convert_annotation_file_to_json, convert_sensor_directory_to_json, fix_timestamps

from trampette_analysis.ml.predict_count import predict_count

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
        annotation_files = list(pathlib.Path(target_path).glob("Annotations*.csv"))
        if annotation_files:
            convert_annotation_file_to_json(annotation_files[0], recording)
        
        # Process each sensor directory
        sensor_dirs = [d for d in pathlib.Path(target_path).iterdir() if d.is_dir()]
        for sensor_dir in sensor_dirs:
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
                if file_path.is_file() and file_path.suffix in {".json", ".mov", '.mp4'}:
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
        
        if "recording_data" in data:
            sensor_data = data["recording_data"]
            timestamps = sensor_data["timestamp"]

            fixed_timestamps = fix_timestamps(timestamps)
            data["recording_data"]["timestamp"] = fixed_timestamps

            # Save the updated JSON data back to the file
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=4)
    
    jump_count = 0
    sensor_mapping = {
        "203130000371": "chest",
        "233830000584": "lower_back",
        "233830000632": "thigh",
        "233830000582": "shin",

        "244730001997": "chest",
        "244730001964": "lower_back",
        "244730001978": "thigh",
        "244730001982": "shin",

        "244730001955": "chest",
        "244730001986": "lower_back",
        "244730001966": "thigh",
        "240530000124": "shin"
    }

    dfs = []
    for json_file in pathlib.Path(target_path).rglob("*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)

        sensor_id = os.path.splitext(os.path.basename(json_file))[0]
        if sensor_id not in sensor_mapping:
            print(f"Skipping unrecognized sensor file: {json_file}")
            continue

        data_dict = data["recording_data"]
        placement = sensor_mapping[sensor_id]
        selected_columns = ['timestamp', 'accx', 'accy', 'accz', 'gyrox', 'gyroy', 'gyroz', 'magnx', 'magny', 'magnz']

        if not all(col in data_dict for col in selected_columns):
            missing = [col for col in selected_columns if col not in data_dict]
            raise ValueError(f"Missing columns in recording_data: {missing}")

        # Find the minimum length of all selected columns
        lengths = [len(data_dict[col]) for col in selected_columns]
        min_len = min(lengths)

        # Trim all columns to the minimum length
        trimmed_data = {col: data_dict[col][:min_len] for col in selected_columns}

        # Create DataFrame from trimmed data
        df = pd.DataFrame(trimmed_data)
        df["timestamp"] = pd.to_datetime(
            df["timestamp"].astype(float),
            unit="s",
            origin="unix"
        )
        # Rename columns
        df.rename(columns={
            'accx': f'{placement}_accx',
            'accy': f'{placement}_accy',
            'accz': f'{placement}_accz',
            'gyrox': f'{placement}_gyrox',
            'gyroy': f'{placement}_gyroy',
            'gyroz': f'{placement}_gyroz',
            'magnx': f'{placement}_magnx',
            'magny': f'{placement}_magny',
            'magnz': f'{placement}_magnz'
        }, inplace=True)

        dfs.append(df)

    # Combine all dataframes
    if dfs:
        start_times = [df['timestamp'].min() for df in dfs]
        end_times = [df['timestamp'].max() for df in dfs]        
        latest_start = max(start_times)
        earliest_end = min(end_times)
        print(f"latest start = {latest_start} and earliest end = {earliest_end}")
        trimmed_dfs = []
        for df in dfs:
            df_trimmed = df[(df['timestamp'] >= latest_start) & (df['timestamp'] <= earliest_end)]
            trimmed_dfs.append(df_trimmed)

        df_merged = trimmed_dfs[0]
        for sensor_df in trimmed_dfs[1:]:
            df_merged = pd.merge_asof(df_merged, sensor_df, on="timestamp", direction="nearest")
    else:
        df_merged = pd.DataFrame()
        print("No valid sensor data found.")

    jump_count = predict_count(df_merged)
        
    # Write the jump count into the files
    for json_file in pathlib.Path(target_path).rglob("*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)
        if "recording_info" in data:
            data["recording_info"]["jump_count"] = jump_count

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
