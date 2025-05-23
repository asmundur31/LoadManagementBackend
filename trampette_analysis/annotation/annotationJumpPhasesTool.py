import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
import csv
import cv2

def load_json(file_path: str) -> pd.DataFrame:
    with open(file_path, 'r') as f:
        data = json.load(f)

    rec_data = data.get('recording_data', {})
    if isinstance(rec_data, dict):
        keys_to_exclude = ['average', 'rrdata']
        for key in keys_to_exclude:
            rec_data.pop(key, None)

        timestamps = rec_data.get('timestamp')
        if timestamps is None or not isinstance(timestamps, list):
            print("Error: 'timestamp' key is missing or not a list in recording_data.")
            return pd.DataFrame()

        desired_length = len(timestamps)
        for key, value in rec_data.items():
            if isinstance(value, list):
                rec_data[key] = value[:desired_length]

        return pd.DataFrame(rec_data)
    return pd.DataFrame()

def save_annotations():
    """Save all annotations to CSV."""
    global annotations_path
    with open(annotations_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Landing type", "Timestamp (ms)", "Phase"])
        writer.writerows(annotations)

# Path to the CSV file containing sensor placement information
sensor_placement_csv_path = "/Users/asmundur/Developer/MasterThesis/data/annotations/sensorPlacement.csv"
sensor_placement_df = pd.read_csv(sensor_placement_csv_path)

# Load IMU Data
subject_id = 7
data_type = "acc"
sensor_placement = "lower_back"
landing_type = "Soft"


# Find the correct sensor name based on subject_id and sensor_placement
sensor_name = sensor_placement_df.loc[
    (sensor_placement_df['Subject Id'] == subject_id) & 
    (sensor_placement_df['Placement'] == sensor_placement),
    'Sensor'
].values[0]

# Build the IMU data path dynamically
imu_data_path = f"/Users/asmundur/Developer/MasterThesis/data_public/raw/Protocol data/{subject_id}/{sensor_name}.json"
video_start_path = f"/Users/asmundur/Developer/MasterThesis/data/annotations/VideoStartTime.csv"

df = load_json(imu_data_path)

# Load video
df_video_start = pd.read_csv(video_start_path)
video_start_data = df_video_start[
    (df_video_start["Landing type"] == landing_type) &
    (df_video_start["Subject Id"] == subject_id)
]
video_path = video_start_data["Video"].values[0]  # Get video path
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
video_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps > 0 else 0

# User-defined video start time (adjustable)
video_start_offset = float(video_start_data["Timestamp (s)"].values[0])

# Load annotations
annotations = []
annotations_path = f"/Users/asmundur/Developer/MasterThesis/data/annotations/{subject_id}/JumpPhases.csv"
if os.path.isfile(annotations_path):
    with open(annotations_path, mode="r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        annotations = [(row[0], float(row[1]), row[2]) for row in reader]

timestamps = df["timestamp"]
print(timestamps[0])
acc_x = df[f"{data_type}x"]
acc_y = df[f"{data_type}y"]
acc_z = df[f"{data_type}z"]

window_size = 10
max_window_size = 30
min_window_size = 1
current_start = video_start_offset

# Phase labels (keys: 1-8)
phase_labels = {
    "1": "Static Phase Start",
    "2": "Run-up Phase Start",
    "3": "Injump Phase Start",
    "4": "Loading Phase Start",
    "5": "Rebound Phase Start",
    "6": "Flight Phase Start",
    "7": "Landing Phase Start",
    "8": "Landing Phase End",
}

fig, ax = plt.subplots(figsize=(10, 5))

# OpenCV video window
cv2.namedWindow("Video", cv2.WINDOW_NORMAL)

def update_video(timestamp):
    """Update video frame to match the given timestamp."""
    if cap.isOpened():
        video_time = timestamp - video_start_offset  # Convert to seconds and adjust offset
        cap.set(cv2.CAP_PROP_POS_MSEC, video_time*1000)  # Seek to new timestamp

        success, frame = cap.read()
        if success:
            frame = cv2.resize(frame, (640, 360))  # Resize for display
            cv2.imshow("Video", frame)
            cv2.waitKey(1)  # Refresh video frame


def on_mouse_move(event):
    """Update the video based on mouse position over the plot."""
    if event.xdata is not None:
        timestamp = float(event.xdata)
        update_video(timestamp)


def update_plot():
    """Update the IMU plot and sync video."""
    global current_start, window_size
    current_end = current_start + window_size
    mask = (timestamps >= current_start) & (timestamps <= current_end)

    ax.clear()
    ax.set_xlabel("Time (s)")
    if data_type == "acc":
        ax.set_ylabel("Acceleration (m/s²)")
    elif data_type == "gyro":
        ax.set_ylabel("Angular velocity (°/s)")
    else:
        ax.set_ylabel("Magnetic field (µT)")

    ax.set_title("IMU Data & Video")
    ax.plot(timestamps[mask], acc_x[mask], label="X-axis", color="r")
    ax.plot(timestamps[mask], acc_y[mask], label="Y-axis", color="g")
    ax.plot(timestamps[mask], acc_z[mask], label="Z-axis", color="b")

    for (landing_type, time, label) in annotations:
        if current_start <= time <= current_end:
            ax.axvline(time, color="k", linestyle="--")
            #ax.text(time, np.max([acc_x, acc_y, acc_z]), label, color="black", rotation=45)

    ax.set_xlim(current_start, current_end)
    ax.set_ylim(min(np.min(acc_x), np.min(acc_y), np.min(acc_z)), 
                max(np.max(acc_x), np.max(acc_y), np.max(acc_z)))
    ax.legend()
    plt.draw()


def on_key(event):
    """Handle keyboard inputs for IMU scrolling, annotation, and video adjustment."""
    global current_start, annotations, window_size, video_start_offset

    if event.key == "right":
        current_start += int(window_size * 0.8)
    elif event.key == "left":
        current_start = max(0, current_start - int(window_size * 0.8))
    elif event.key == "up":
        window_size = min(window_size + 1, max_window_size)
    elif event.key == "down":
        window_size = max(window_size - 1, min_window_size)
    elif event.key == "+":
        video_start_offset += 0.1  # Adjust video sync (move video forward)
        print(f"Video Start Offset: {video_start_offset:.4f} sec")
    elif event.key == "-":
        video_start_offset -= 0.1  # Adjust video sync (move video backward)
        print(f"Video Start Offset: {video_start_offset:.4f} sec")
    elif event.key in phase_labels and event.xdata is not None:
        timestamp = float(event.xdata)
        annotations.append((landing_type, timestamp, phase_labels[event.key]))
        save_annotations()
    elif event.key == "u" and annotations:
        removed = annotations.pop()
        save_annotations()

    update_plot()

update_plot()
fig.canvas.mpl_connect("key_press_event", on_key)
# Connect mouse movement to video update
fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)
plt.show()

cap.release()
cv2.destroyAllWindows()