import pandas as pd
import json
import os
import glob


def load_jump_annotaitons(base_path: str = "data_public") -> pd.DataFrame:
    jump_annotations_path = os.path.join(base_path, "annotations", "jumpAnnotations.csv")
    all_jump_annotations_df = pd.read_csv(jump_annotations_path)
    return all_jump_annotations_df 

def load_imu_data_and_annotations(subject_id: int, base_path: str = "data_public") -> tuple:
    """
    Load IMU sensor data and annotations for a given subject.

    Parameters:
    - subject_id (int): Subject folder name
    - base_path (str): Path to the dataset root

    Returns:
    - sensors_df (pd.DataFrame): IMU data with timestamps
    - jump_phases_df (pd.DataFrame): Annotated jump phases with timestamps
    """
    # Paths to the data files
    sensor_files = glob.glob(f"{base_path}/raw/Protocol data/{subject_id}/*.json")
    sensor_placement_path = os.path.join(base_path, "annotations", "sensorPlacement.csv")
    sensor_placement_df = pd.read_csv(sensor_placement_path)
    
    # Load sensor data
    sensor_dfs = []  # List to store each sensor's DataFrame
    min_timestamp = None  # Track the earliest start time
    max_timestamp = None  # Track the earliest stop time

    for file in sensor_files:
        with open(file, "r") as f:
            data = json.load(f)

        # Extract data
        sensor_data = data["recording_data"]
        sensor_df = pd.DataFrame(sensor_data)
        sensor_df["timestamp"] = pd.to_datetime(
            sensor_df["timestamp"].astype(float),
            unit="s",
            origin="unix"
        )

        sensor_name = int(file.split("/")[-1].replace(".json", ""))  # Extract filename

        sensor_placement = sensor_placement_df.loc[
            (sensor_placement_df['Subject Id'] == subject_id) & 
            (sensor_placement_df['Sensor'] == sensor_name),
            'Placement'
        ].values[0]

        # Track the earliest start time
        if min_timestamp is None or sensor_df["timestamp"].min() < min_timestamp:
            min_timestamp = sensor_df["timestamp"].min()
        
        # Track the earliest stop time (shortest recording)
        if max_timestamp is None or sensor_df["timestamp"].max() < max_timestamp:
            max_timestamp = sensor_df["timestamp"].max()

        # Rename columns with body placement
        for col in sensor_df.columns:
            if col != "timestamp":
                sensor_df.rename(columns={col: f"{sensor_placement}_{col}"}, inplace=True)

        sensor_dfs.append(sensor_df)


    # Cut off all recordings at the earliest stop time
    sensor_dfs = [df[df["timestamp"] <= max_timestamp] for df in sensor_dfs]

    # Merge all sensor DataFrames on timestamp
    df = sensor_dfs[0]
    for sensor_df in sensor_dfs[1:]:
        df = pd.merge_asof(df, sensor_df, on="timestamp", direction="nearest")

    sensors_df = df.sort_values("timestamp").reset_index(drop=True)

    # Load annotation data
    jump_phases_path = os.path.join(base_path, "annotations", str(subject_id), "JumpPhases.csv")
    jump_phases_df = pd.read_csv(jump_phases_path)
    jump_phases_df["Timestamp (s)"] = pd.to_datetime(
        jump_phases_df["Timestamp (s)"].astype(float),
        unit="s", 
        origin="unix"
    )
    jump_phases_df = jump_phases_df.sort_values("Timestamp (s)").reset_index(drop=True)
    # Load jump annotations
    jump_annotations_path = os.path.join(base_path, "annotations", "jumpAnnotations.csv")
    all_jump_annotations_df = pd.read_csv(jump_annotations_path)
    jump_annotations_df = all_jump_annotations_df.loc[all_jump_annotations_df['Subject ID'] == subject_id]
    return sensors_df, jump_phases_df, jump_annotations_df

def combine_imu_data_and_annotations(sensor_df: pd.DataFrame, annotation_df: pd.DataFrame) -> pd.DataFrame:
    # Define phase names and their corresponding integer labels
    phase_order = {
        "Static Phase Start": 1,
        "Run-up Phase Start": 2,
        "Injump Phase Start": 3,
        "Loading Phase Start": 4,
        "Rebound Phase Start": 5,
        "Flight Phase Start": 6,
        "Landing Phase Start": 7
    }

    # Sort annotations by timestamp
    annotation_df = annotation_df.sort_values("Timestamp (s)").reset_index(drop=True)

    # Create a list to store labeled intervals
    intervals = []
    labels = []

    # Loop through the annotation rows pairwise to define intervals
    for i in range(len(annotation_df) - 1):
        phase_name = annotation_df.loc[i, 'Phase']
        start = annotation_df.loc[i, 'Timestamp (s)']
        end = annotation_df.loc[i + 1, 'Timestamp (s)']

        label = phase_order.get(phase_name)
        if label is not None:
            intervals.append(pd.Interval(left=start, right=end, closed='left'))
            labels.append(label)
    # Build a label map from intervals
    label_map = pd.Series(labels, index=pd.IntervalIndex(intervals))

    # Assign labels to timestamps
    sensor_df["label"] = sensor_df["timestamp"].map(label_map).fillna(0).astype(int)

    return sensor_df

def get_jump(sensor_df: pd.DataFrame, jump_df:pd.DataFrame, jump_nr: int) -> pd.DataFrame:
    """
    Get the jump data for a specific jump number.
    
    Parameters:
    - sensor_df (pd.DataFrame): DataFrame containing sensor data with timestamps and labels
    - jump_nr (int): Jump number to extract
    
    Returns:
    - pd.DataFrame: DataFrame containing the data for the specified jump
    """
    jump = jump_df.loc[jump_df['Jump Nr'] == jump_nr]
    if jump.empty:
        print(f"Jump number {jump_nr} not found in the jump DataFrame.")
        return pd.DataFrame(), pd.DataFrame()
    print(jump[["Jump Nr", "Subject ID", "Landing", "Trial", "Skill", "Video ToF (s)", "ToF Height (m)", "Height from floor (m)", "Adjusted FSS (m/s)"]])
    
    is_jump = sensor_df['label'] != 0
    jump_segments = []
    in_jump = False
    start_idx = None

    for idx, val in is_jump.items():
        if val and not in_jump:
            # Start of a jump
            start_idx = idx
            in_jump = True
        elif not val and in_jump:
            # End of a jump
            end_idx = idx
            jump_segments.append((start_idx, end_idx))
            in_jump = False

    # If the data ends while still in a jump
    if in_jump and start_idx is not None:
        jump_segments.append((start_idx, sensor_df.index[-1] + 1))
    
    # Extract the desired jump
    start, end = jump_segments[jump_nr - 1]
    jump_imu_df = sensor_df.loc[start:end - 1].reset_index(drop=True)
    
    return jump_imu_df, jump

def get_subject_info(subject_id: int, base_path: str = "data_public") -> dict:
    subject_info_path = os.path.join(base_path, "annotations", "subjectInfo.csv")
    subject_info_df = pd.read_csv(subject_info_path)
    
    subject_info = subject_info_df.loc[subject_info_df['Subject Id'] == subject_id]

    return subject_info.to_dict()
