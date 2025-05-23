import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R
from ahrs.filters import Madgwick

def read_json_file(file_path: str) -> dict:
    """
    Reads a JSON file and returns the data as a Python dictionary.
    
    :param file_path: Path to the JSON file.
    :return: Data from the JSON file as a dictionary.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to read JSON file: {e}")

def low_pass_filter(data, alpha=0.9):
    """ Simple low-pass filter to estimate gravity component """
    filtered = [data[0]]  # Initialize with first data point
    for i in range(1, len(data)):
        filtered.append(alpha * filtered[-1] + (1 - alpha) * data[i])
    return filtered

def cumtrapz(y, dx=1.0, initial=0):
    """
    A simple cumulative trapezoidal integration implementation.
    Parameters:
        y (array-like): Input data array.
        dx (float): Spacing between sample points.
        initial (float): The initial value to prepend to the cumulative sum.
    Returns:
        np.ndarray: The cumulative integral of y.
    """
    y = np.asarray(y)
    # Compute the area of each trapezoid: 0.5*(y[i] + y[i+1]) * dx
    trapz_values = np.concatenate(([initial], np.cumsum(0.5 * (y[:-1] + y[1:]) * dx)))
    return trapz_values

def butter_lowpass(cutoff, fs, order=4):
    """Creates a low-pass Butterworth filter."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_highpass(cutoff, fs, order=4):
    """Creates a high-pass Butterworth filter."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def compute_orientation_madgwick(acc_data, gyro_data, mag_data, dt, beta=0.1):
    """
    Compute orientation using the Madgwick filter. Returns rotation matrices.
    This step converts IMU data to useful orientation information.
    
    Parameters:
    -----------
    acc_data : np.ndarray, shape (N,3)
        Accelerometer data (m/s²).
    gyro_data : np.ndarray, shape (N,3)
        Gyroscope data (rad/s).
    mag_data : np.ndarray, shape (N,3)
        Magnetometer data.
    dt : float
        Sampling interval (seconds).
    beta : float, optional
        Madgwick filter gain.

    Returns:
    --------
    rotation_matrices : np.ndarray, shape (N, 3, 3)
        Rotation matrices that transform sensor-frame acceleration to the global frame.
    """
    from ahrs.filters import Madgwick  # Requires `pip install ahrs`
    
    madgwick = Madgwick(beta=beta, frequency=1/dt)
    q = np.zeros((acc_data.shape[0], 4))  # Quaternion storage
    q[0] = [1, 0, 0, 0]  # Initial quaternion

    gyro_data = np.deg2rad(gyro_data)
    
    for i in range(1, len(acc_data)):
      q[i] = madgwick.updateMARG(q[i-1], gyr=gyro_data[i], acc=acc_data[i], mag=mag_data[i])
    
    # Convert quaternion to rotation matrices
    rotation_matrices = np.array([R.from_quat(q[i]).as_matrix() for i in range(len(q))])
    
    return rotation_matrices

def process_acceleration_data(acc_data, gyro_data, mag_data, dt, gravity=9.81, lp_fc=5.0, hp_fc=0.1, zupt_threshold_factor=0.5):
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
    gravity_vector = np.array([0, 0, -gravity])  # Gravity in global frame
    acc_corrected = np.zeros_like(acc_data)

    for i in range(acc_data.shape[0]):
        gravity_in_sensor_frame = rotation_matrices[i] @ gravity_vector
        acc_corrected[i] = acc_data[i] - gravity_in_sensor_frame

    # Step 3: Compute acceleration magnitude (ignoring orientation)
    acc_magnitude = np.linalg.norm(acc_corrected, axis=1)  # |a| = sqrt(ax² + ay² + az²)

    # Step 4: Apply low-pass filter to remove noise
    fs = 1.0 / dt  # Sampling frequency (Hz)
    b_lp, a_lp = butter_lowpass(lp_fc, fs)
    acc_filtered = filtfilt(b_lp, a_lp, acc_magnitude)

    # Step 5: Apply high-pass filter to remove drift
    b_hp, a_hp = butter_highpass(hp_fc, fs)
    acc_filtered = filtfilt(b_hp, a_hp, acc_filtered)

    # Step 6: Integrate acceleration to obtain velocity
    velocity = cumtrapz(acc_filtered, dx=dt, initial=0)

    # Step 8: Ensure velocity starts at zero
    velocity -= velocity[0]

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
start_seconds = 981.3
end_seconds = 984.8
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
plt.figure(figsize=(10, 6))
plt.plot(filtered_time_intervals, filtered_acceleration_data['accX'], label='Acceleration X')
plt.plot(filtered_time_intervals, filtered_acceleration_data['accY'], label='Acceleration Y')
plt.plot(filtered_time_intervals, filtered_acceleration_data['accZ'], label='Acceleration Z')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s²)')
plt.title('Acceleration Data Section')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(filtered_time_intervals, filtered_gyro_data['gyroX'], label='Gyropscope X')
plt.plot(filtered_time_intervals, filtered_gyro_data['gyroY'], label='Gyropscope Y')
plt.plot(filtered_time_intervals, filtered_gyro_data['gyroZ'], label='Gyropscope Z')
plt.xlabel('Time (s)')
plt.ylabel('Gyroscope (degree)')
plt.title('Gyroscope Data Section')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(filtered_time_intervals, filtered_mag_data['magnX'], label='Megnetometer X')
plt.plot(filtered_time_intervals, filtered_mag_data['magnY'], label='Megnetometer Y')
plt.plot(filtered_time_intervals, filtered_mag_data['magnZ'], label='Megnetometer Z')
plt.xlabel('Time (s)')
plt.ylabel('Magnetometer (uT)')
plt.title('Magnetometer Data Section')
plt.legend()
plt.show()

# --- 5. Process the acceleration data to obtain velocity and displacement ---
# Here, we assume the run direction is along the x-axis. Adjust if needed.
print(dt)
velocities, displacements = process_acceleration_data(acc_array, gyro_array, mag_array, dt)
print(velocities.shape)
print(displacements.shape)

# --- 6. Plot the linear velocity over time ---
plt.figure(figsize=(10, 6))
plt.plot(filtered_time_intervals, velocities, label='Linear Velocity (x-direction)')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Linear Velocity over Time (Section)')
plt.legend()
plt.show()

# (Optional) Print final values:
print("Final velocity (m/s):", velocities[-1])
print("Max velocity (m/s):", np.max(velocities))
print("Total displacement (m):", displacements[-1])