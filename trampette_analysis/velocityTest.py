import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from ahrs.filters import Madgwick  # pip install ahrs
from utility import read_json_file

def compute_velocity_displacement(timestamps, acc, gyro, mag, static_duration=3.0, gravity=9.81):
    """
    Estimates linear velocity and displacement from IMU data.
    
    This version computes the sensor bias (without gravity) during the static period.
    
    Parameters:
      timestamps      : numpy array of timestamps (seconds) of shape (N,)
      acc             : numpy array of accelerometer data (N x 3, in m/s², sensor frame)
      gyro            : numpy array of gyroscope data (N x 3, in deg/s, sensor frame)
      mag             : numpy array of magnetometer data (N x 3)
      static_duration : Duration (in seconds) at start during which the sensor is assumed static.
      gravity         : Gravitational acceleration (m/s²)
      
    Returns:
      velocity         : Estimated velocities (N x 3, in m/s, global frame)
      position         : Estimated positions (N x 3, in m, global frame)
      rotation_matrices: Rotation matrices (N x 3 x 3) mapping sensor -> global
    """
    # ---------------------------
    # Step 1: Static Calibration (Bias Without Gravity)
    # ---------------------------
    static_mask = (timestamps - timestamps[0]) < static_duration
    acc_static = acc[static_mask]
    gyro_static = gyro[static_mask]
    
    # Compute the static average accelerometer reading in sensor frame.
    static_avg = np.mean(acc_static, axis=0)
    
    # Get the sensor's static orientation by running the Madgwick filter over the static period.
    dt = np.mean(np.diff(timestamps))
    madgwick_static = Madgwick(dt=dt)
    q_static = np.array([1.0, 0.0, 0.0, 0.0])
    for i in range(np.sum(static_mask)):
        q_static = madgwick_static.updateMARG(q_static,
                                              gyr=np.deg2rad(gyro_static[i]),
                                              acc=acc_static[i],
                                              mag=mag[static_mask][i])
    # Convert q_static to rotation matrix (using SciPy convention: [q1,q2,q3,q0])
    quat_scipy = np.array([q_static[1], q_static[2], q_static[3], q_static[0]])
    R_static = R.from_quat(quat_scipy).as_matrix()
    # Define the global gravity vector (here we assume global Z-axis is up)
    gravity_global = np.array([0, 0, gravity])
    # Transform global gravity into sensor frame (R_static maps sensor->global so transpose gives global->sensor)
    gravity_sensor = R_static.T @ gravity_global
    # Sensor bias (excluding gravity) is the difference between the static average and expected gravity in sensor frame.
    bias = static_avg - gravity_sensor
    print("Static average (sensor frame):", static_avg)
    print("Expected gravity in sensor frame:", gravity_sensor)
    print("Estimated sensor bias (excluding gravity):", bias)
    
    # ---------------------------
    # Step 2: Bias Correction
    # ---------------------------
    acc_corr = acc - bias
    gyro_bias = np.mean(gyro_static, axis=0)
    gyro_corr = gyro - gyro_bias
    gyro_corr = np.deg2rad(gyro_corr)  # convert to rad/s
    
    # ---------------------------
    # Step 3: Orientation Estimation via Madgwick Filter
    # ---------------------------
    N = len(timestamps)
    madgwick = Madgwick(dt=dt)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    orientation_quat = np.zeros((N, 4))
    rotation_matrices = np.zeros((N, 3, 3))
    for i in range(N):
        q = madgwick.updateMARG(q, gyr=gyro_corr[i], acc=acc_corr[i], mag=mag[i])
        orientation_quat[i] = q
        quat_scipy = np.array([q[1], q[2], q[3], q[0]])
        rotation_matrices[i] = R.from_quat(quat_scipy).as_matrix()
    
    # ---------------------------
    # Step 4: Transform Acceleration to Global Frame and Remove Gravity
    # ---------------------------
    acc_global = np.zeros_like(acc_corr)
    for i in range(N):
        # rotation_matrices[i] maps sensor->global.
        acc_global[i] = rotation_matrices[i] @ acc_corr[i]
    # Remove the known gravity vector (global)
    acc_dynamic = acc_global - gravity_global
    
    # ---------------------------
    # Step 5: Numerical Integration
    # ---------------------------
    velocity = np.zeros_like(acc_dynamic)
    position = np.zeros_like(acc_dynamic)
    for i in range(1, N):
        velocity[i] = velocity[i-1] + 0.5 * (acc_dynamic[i] + acc_dynamic[i-1]) * dt
        position[i] = position[i-1] + 0.5 * (velocity[i] + velocity[i-1]) * dt
    
    # Enforce zero velocity/position during static period.
    velocity[static_mask] = 0
    position[static_mask] = 0
    
    return velocity, position, rotation_matrices, bias, gravity_global, acc, acc_corr, acc_global, acc_dynamic


# Example usage:
if __name__ == "__main__":
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
    start_seconds = 212.9 #981.3
    end_seconds = 219.1 #984.8
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
 
    velocity, position, rotation_matrices, bias, gravity_global, raw_acc, acc_bias_corr, acc_global, acc_dynamic = compute_velocity_displacement(
        filtered_time_intervals, acc_array, gyro_array, mag_array, static_duration=3.0
    )

    # Compute norms for plotting:
    raw_acc_norm = np.linalg.norm(raw_acc, axis=1)
    bias_corr_norm = np.linalg.norm(acc_bias_corr, axis=1)
    dynamic_acc_norm = np.linalg.norm(acc_dynamic, axis=1)
    
    # Plot acceleration norms, velocity, and position.
    time_vector = filtered_time_intervals - filtered_time_intervals[0]
    fig, axs = plt.subplots(5, 1, figsize=(10, 16), sharex=True)
    
    # Raw acceleration norm.
    axs[0].plot(time_vector, raw_acc[:,0], label='Acc X')
    axs[0].plot(time_vector, raw_acc[:,1], label='Acc Y')
    axs[0].plot(time_vector, raw_acc[:,2], label='Acc Z')
    axs[0].set_title("Raw Accelerometer Data")
    axs[0].set_ylabel("Acc (m/s²)")
    axs[0].grid(True)
    
    # Bias-corrected acceleration norm (sensor frame).
    axs[1].plot(time_vector, acc_bias_corr[:,0], label='Acc X')
    axs[1].plot(time_vector, acc_bias_corr[:,1], label='Acc Y')
    axs[1].plot(time_vector, acc_bias_corr[:,2], label='Acc Z')
    axs[1].set_title("Bias-Corrected Accelerometer Data (Sensor Frame)")
    axs[1].set_ylabel("Acc (m/s²)")
    axs[1].grid(True)
    
    # Global acceleration norm (after bias correction).
    acc_global_norm = np.linalg.norm(acc_global, axis=1)
    axs[2].plot(time_vector, acc_global[:,0], label='Acc X')
    axs[2].plot(time_vector, acc_global[:,1], label='Acc Y')
    axs[2].plot(time_vector, acc_global[:,2], label='Acc Z')
    axs[2].set_title("Global Accelerometer Data (After Bias Correction)")
    axs[2].set_ylabel("Acc (m/s²)")
    axs[2].grid(True)
    
    # Dynamic acceleration norm (global, gravity removed).
    axs[3].plot(time_vector, acc_dynamic[:,0], label='Acc X')
    axs[3].plot(time_vector, acc_dynamic[:,1], label='Acc Y')
    axs[3].plot(time_vector, acc_dynamic[:,2], label='Acc Z')
    axs[3].set_title("Dynamic Acceleration (Global, Gravity Removed)")
    axs[3].set_ylabel("Acc (m/s²)")
    axs[3].grid(True)
    
    # Velocity (plotting X, Y, Z components)
    axs[4].plot(time_vector, velocity[:,0], label='Vel X')
    axs[4].plot(time_vector, velocity[:,1], label='Vel Y')
    axs[4].plot(time_vector, velocity[:,2], label='Vel Z')
    axs[4].set_title("Estimated Velocity (Global)")
    axs[4].set_ylabel("Velocity (m/s)")
    axs[4].set_xlabel("Time (s)")
    axs[4].legend()
    axs[4].grid(True)
    
    plt.tight_layout()
    plt.show()

    # Optional: You can also plot position similarly.
    fig2, axs2 = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axs2[0].plot(time_vector, position[:,0], label='Pos X')
    axs2[1].plot(time_vector, position[:,1], label='Pos Y')
    axs2[2].plot(time_vector, position[:,2], label='Pos Z')
    for ax in axs2:
        ax.grid(True)
        ax.legend()
    axs2[2].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()