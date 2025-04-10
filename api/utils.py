'''
    This module is for utility functions.
'''
import csv
import json
import shutil
from pathlib import Path

def convert_annotation_file_to_json(csv_file: Path, recording: dict):
    json_file_path = csv_file.with_suffix('.json')
    with open(csv_file, 'r') as csv_f, open(json_file_path, 'w') as json_f:
        reader = csv.DictReader(csv_f)
        headers = reader.fieldnames  # Get the headers from the CSV file
        data = {header.lower(): [] for header in headers}  # Initialize the data dictionary with headers as keys
        for row in reader:
            for header in headers:
                if header == "Timestamp":
                    data[header.lower()].append(float(row[header]))
                else:
                    data[header.lower()].append(row[header])
        json_data = {
            "recording_info": {
                "recording_id": str(recording["id"]),
                "recording_name": recording["recording_name"],
                "user_id": str(recording["user_id"]),
                "frequency": 1/104.0
            },
            "recording_data": data
        }
        json.dump(json_data, json_f, indent=4)  # Write JSON
        csv_file.unlink()

def convert_sensor_directory_to_json(directory_path: Path, recording: dict):
    data = {}
    timestamp_added = False
    for csv_file in directory_path.glob("*.csv"):
        with open(csv_file, 'r') as csv_f:
            next(csv_f) # Skip the first header line
            reader = csv.DictReader(csv_f)
            headers = reader.fieldnames
            for header in headers:
                if header.lower() not in data:
                    data[header.lower()] = []
            
            rows = list(reader)
            for row in rows[:-4]: # Process all but the last 4 rows
                for header in headers:
                    if header == "Timestamp":
                        if not timestamp_added:
                            data[header.lower()].append(float(row[header]))
                    else:
                        data[header.lower()].append(float(row[header]))
            if "timestamp" in data:
                timestamp_added = True
            csv_file.unlink()
    
    json_data = {
        "recording_info": {
            "recording_id": str(recording["id"]),
            "recording_name": recording["recording_name"],
            "user_id": str(recording["user_id"]),
            "sensor_name": directory_path.name,
            "frequency": 1/104.0
            },
        "recording_data": data
    }

    json_file_path = directory_path.with_suffix('.json')
    with open(json_file_path, 'w') as json_f:
        json.dump(json_data, json_f, indent=4)
    shutil.rmtree(directory_path)

def fix_timestamps(timestamps: list):
    """
        Function to fix timestamps.
    """
    if len(timestamps) <= 1:
        return timestamps

    fixed_timestamps = timestamps.copy()
    j = 0
    for i in range(1, len(timestamps)):
        if timestamps[i] != timestamps[i - 1]:
            n = i - j  # Number of datapoints
            frequencyNow = (timestamps[i] - timestamps[i - 1]) / n
            j = i - n
            while j < i - 1:
                fixed_timestamps[j] -= (float((i-j-1))*frequencyNow)
                j += 1
            j += 1

    return fixed_timestamps