'''
    This module is for utility functions.
'''
import csv
import json
from pathlib import Path

from api.models import Recording

def convert_csv_file_to_json(csv_file: Path, recording: dict):
    json_file_path = csv_file.with_suffix('.json')
    with open(csv_file, 'r') as csv_f, open(json_file_path, 'w') as json_f:
        if not "Annotations" in csv_file.name:
            # Skip the first line for non-annotation files
            next(csv_f)
        reader = csv.DictReader(csv_f)
        headers = reader.fieldnames  # Get the headers from the CSV file
        data = {header: [] for header in headers}  # Initialize the data dictionary with headers as keys
        for row in reader:
            for header in headers:
                data[header].append(row[header])
        json_data = {
            "recording_info": {
                "recording_id": recording["id"],
                "recording_name": recording["recording_name"],
                "user_id": recording["user_id"]
            },
            "data": data
        }
        json.dump(json_data, json_f, indent=4)  # Write JSON