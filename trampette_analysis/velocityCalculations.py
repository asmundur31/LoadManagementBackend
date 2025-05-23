import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import filtfilt
from utility import (
    compute_orientation_madgwick, 
    butter_lowpass, 
    butter_highpass, 
    cumtrapz, 
    read_json_file, 
    animate_orientation, 
    plot_xyz_data, 
    plot_data
)

def process_acceleration_data(time_data, acc_data, gyro_data, mag_data, dt, gravity=9.81, lp_fc=5.0, hp_fc=0.1, zupt_threshold_factor=0.5):
    """
    Process IMU acceleration data using sensor fusion to estimate velocity and displacement.

    Parameters:
    -----------
    acc_data : np.ndarray, shape (N,3)
      Raw acceleration data in m/s².
    gyro_data : np.ndarray, shape (N,3)
      Gyroscope data (rad/s).
    mag_data : np.ndarray, shape (N,3)
      Magnetometer data.
    dt : float
      Sampling interval in seconds.
    gravity : float, optional
      Gravitational acceleration (default 9.81 m/s²).
    lp_fc : float, optional
      Cutoff frequency (Hz) for the low-pass filter (for noise reduction).
    hp_fc : float, optional
      Cutoff frequency (Hz) for the high-pass filter (for drift correction).
    zupt_threshold_factor : float, optional
      Multiplier for adaptive motion threshold based on standard deviation.

    Returns:
    --------
    velocity : np.ndarray, shape (N,)
      Corrected velocity over time.
    displacement : np.ndarray, shape (N,)
      Corrected displacement over time.
    """
    # Step 1: Compute orientation from sensor fusion (Madgwick filter)
    rotation_matrices = compute_orientation_madgwick(acc_data, gyro_data, mag_data, dt)
    # Step 2: Rotate gravity vector to sensor frame and remove from acceleration
    gravity_global = np.array([0, -gravity, 0])  # Gravity in global frame
    acc_global = np.zeros_like(acc_data)
    acc_corrected_global = np.zeros_like(acc_data)

    for i in range(acc_data.shape[0]):
        acc_global[i] = rotation_matrices[i] @ acc_data[i]
        acc_corrected_global[i] = acc_global[i] - gravity_global
    
    animate_orientation(rotation_matrices, acc_corrected_global)
    plot_xyz_data(
        time_data,
        acc_corrected_global[:,0],
        acc_corrected_global[:,1],
        acc_corrected_global[:,2],
        title="Corrected Acceleration Data",
        labels=("X", "Y", "Z")
    )
    # Step 3: Compute acceleration magnitude (ignoring orientation)
    #acc_magnitude = np.linalg.norm(acc_corrected_global, axis=1)  # |a| = sqrt(ax² + ay² + az²)

    # Step 4: Apply low-pass filter to remove noise
    fs = 1.0 / dt  # Sampling frequency (Hz)
    b_lp, a_lp = butter_lowpass(lp_fc, fs)
    acc_filtered = filtfilt(b_lp, a_lp, acc_corrected_global[:,2])

    # Step 5: Apply high-pass filter to remove drift
    b_hp, a_hp = butter_highpass(hp_fc, fs)
    acc_filtered = filtfilt(b_hp, a_hp, acc_filtered)

    # Step 6: Integrate acceleration to obtain velocity
    velocity = cumtrapz(acc_filtered, dx=dt, initial=0)

    # Step 9: Integrate velocity to obtain displacement
    displacement = cumtrapz(velocity, dx=dt, initial=0)

    return velocity, displacement

# Example usage
file_path = "/Users/asmundur/Developer/MasterThesis/data/raw/5/Test with videos/233830000582.json"
data = read_json_file(file_path)
# Convert acceleration data to NumPy arrays.
acceleration_data = {
    'accX': np.array(data["recording_data"]["accx"], dtype=float),
    'accY': np.array(data["recording_data"]["accy"], dtype=float),
    'accZ': np.array(data["recording_data"]["accz"], dtype=float)
}

# Convert gyroscope data to NumPy arrays.
gyro_data = {
    'gyroX': np.array(data["recording_data"]["gyrox"], dtype=float),
    'gyroY': np.array(data["recording_data"]["gyroy"], dtype=float),
    'gyroZ': np.array(data["recording_data"]["gyroz"], dtype=float)
}

# Convert magnetometer data to NumPy arrays.
mag_data = {
    'magnX': np.array(data["recording_data"]["magnx"], dtype=float),
    'magnY': np.array(data["recording_data"]["magny"], dtype=float),
    'magnZ': np.array(data["recording_data"]["magnz"], dtype=float)
}
time_intervals = np.array(data["recording_data"]["timestamp"], dtype=float)
common_length = min(len(time_intervals),
                    len(acceleration_data['accX']),
                    len(acceleration_data['accY']),
                    len(acceleration_data['accZ']),
                    len(gyro_data['gyroX']),
                    len(gyro_data['gyroY']),
                    len(gyro_data['gyroZ']),
                    len(mag_data['magnX']),
                    len(mag_data['magnY']),
                    len(mag_data['magnZ']))  
time_intervals = time_intervals[:common_length]
for key in acceleration_data:
    acceleration_data[key] = acceleration_data[key][:common_length]
for key in gyro_data:
    gyro_data[key] = gyro_data[key][:common_length]
for key in mag_data:
    mag_data[key] = mag_data[key][:common_length]

# Define start and end seconds for the section
start_seconds = 216.6 #981.3
end_seconds = 219.4 #984.8
# Convert start and end seconds to timestamps
start_time = time_intervals[0] + start_seconds
end_time = time_intervals[0] + end_seconds
# Create a mask
mask = (time_intervals >= start_time) & (time_intervals <= end_time)
filtered_time_intervals = time_intervals[mask]

filtered_acceleration_data = {
    'accX': acceleration_data['accX'][mask],
    'accY': acceleration_data['accY'][mask],
    'accZ': acceleration_data['accZ'][mask]
}
filtered_gyro_data = {
    'gyroX': gyro_data['gyroX'][mask],
    'gyroY': gyro_data['gyroY'][mask],
    'gyroZ': gyro_data['gyroZ'][mask]
}
filtered_mag_data = {
    'magnX': mag_data['magnX'][mask],
    'magnY': mag_data['magnY'][mask],
    'magnZ': mag_data['magnZ'][mask]
}

# Convert the filtered acceleration, gyro, and mag data into Nx3 arrays.
acc_array = np.column_stack((
    filtered_acceleration_data['accX'],
    filtered_acceleration_data['accY'],
    filtered_acceleration_data['accZ']
))
gyro_array = np.column_stack((
    filtered_gyro_data['gyroX'],
    filtered_gyro_data['gyroY'],
    filtered_gyro_data['gyroZ']
))
#gyro_array -= np.mean(gyro_array[:400], axis=0)
mag_array = np.column_stack((
    filtered_mag_data['magnX'],
    filtered_mag_data['magnY'],
    filtered_mag_data['magnZ']
))

# Estimate the sampling interval (assumes uniform sampling).
if len(filtered_time_intervals) > 1:
    dt = np.mean(np.diff(filtered_time_intervals))
else:
    raise ValueError("Not enough time points in the filtered section.")

# --- 4. Plot the filtered acceleration data ---
plot_xyz_data(
    filtered_time_intervals,
    filtered_acceleration_data['accX'],
    filtered_acceleration_data['accY'],
    filtered_acceleration_data['accZ'],
    title="Acceleration Data Section",
    labels=("X", "Y", "Z")
)
plot_xyz_data(
    filtered_time_intervals,
    filtered_gyro_data['gyroX'],
    filtered_gyro_data['gyroY'],
    filtered_gyro_data['gyroZ'],
    title="Gyroscope Data Section",
    labels=("X", "Y", "Z")
)
plot_xyz_data(
    filtered_time_intervals,
    filtered_mag_data['magnX'],
    filtered_mag_data['magnY'],
    filtered_mag_data['magnZ'],
    title="Magnetometer Data Section",
    labels=("X", "Y", "Z")
)


# --- 5. Process the acceleration data to obtain velocity and displacement ---
# Here, we assume the run direction is along the x-axis. Adjust if needed.
print(dt)
velocities, displacements = process_acceleration_data(filtered_time_intervals, acc_array, gyro_array, mag_array, dt)

# --- 6. Plot the linear velocity over time ---
plot_data(
    filtered_time_intervals,
    velocities,
    title="Linear Velocity",
    ylabel="Velocity (m/s)"
)

# (Optional) Print final values:
print("Final velocity (m/s):", velocities[-1])
print("Max velocity (m/s):", np.max(velocities))
print("Total displacement (m):", displacements[-1])