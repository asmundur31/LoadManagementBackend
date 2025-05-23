import json
import numpy as np
from ahrs.filters import Madgwick
from scipy.signal import butter
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

def compute_orientation_madgwick(acc_data, gyro_data, mag_data, dt, beta=0.04):
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
    madgwick = Madgwick(beta=beta, frequency=1/dt)
    q = np.zeros((acc_data.shape[0], 4))  # Quaternion storage
    q[0] = [1, 0, 0, 0]  # Initial quaternion

    gyro_data = np.deg2rad(gyro_data)
    
    for i in range(1, len(acc_data)):
      q[i] = madgwick.updateMARG(q[i-1], gyr=gyro_data[i], acc=acc_data[i], mag=mag_data[i])
    
    # Convert quaternion to rotation matrices
    rotation_matrices = np.array([R.from_quat(q[i]).as_matrix() for i in range(len(q))])
    
    return rotation_matrices


def plot_xyz_data(time, x, y, z, title="Sensor Data", labels=("X", "Y", "Z")):
    """
    Plots X, Y, Z data over time in both 3D and 2D subplots.

    Parameters:
    -----------
    time : np.ndarray
        Time values (N,)
    x : np.ndarray
        X-axis values (N,)
    y : np.ndarray
        Y-axis values (N,)
    z : np.ndarray
        Z-axis values (N,)
    title : str, optional
        Plot title.
    labels : tuple, optional
        Labels for the three axes (default: ("X", "Y", "Z")).
    """
    fig = plt.figure(figsize=(12, 6))

    ax = fig.add_subplot(111)
    ax.plot(time, x, 'r', label=labels[0])
    ax.plot(time, y, 'g', label=labels[1])
    ax.plot(time, z, 'b', label=labels[2])
    ax.set_xlabel("Time")
    ax.set_ylabel("Values")
    ax.set_title(f"{title}")
    ax.legend()
    plt.show()

def plot_data(time, data, title="Sensor Data", ylabel="data"):
    """
    Plots X, Y, Z data over time in both 3D and 2D subplots.

    Parameters:
    -----------
    time : np.ndarray
        Time values (N,)
    data : np.ndarray
        Values (N,)
    title : str, optional
        Plot title.
    labels : tuple, optional
        Labels for the three axes (default: ("data")).
    """
    fig = plt.figure(figsize=(12, 6))

    ax = fig.add_subplot(111)
    ax.plot(time, data, 'blue', label=ylabel)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{ylabel}")
    ax.set_title(f"{title}")
    ax.legend()
    plt.show()


def animate_orientation(rotation_matrices, acc_vectors, freq=104):
    """
    Animates IMU orientation using a sequence of rotation matrices and acceleration vectors.

    Parameters:
      rotation_matrices: List or array of 3x3 rotation matrices.
      acc_vectors: List or array of acceleration vectors (each a 3-element array).
      freq: Frame rate in Hz.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.clear()
        # Get current rotation matrix and acceleration vector
        R_current = rotation_matrices[frame]
        acc = acc_vectors[frame]  # expected shape (3,)
        
        # Define IMU axes (unit vectors in sensor frame)
        origin = np.array([[0, 0, 0]]).T
        x_axis = np.array([[10, 0, 0]]).T
        y_axis = np.array([[0, 10, 0]]).T
        z_axis = np.array([[0, 0, 10]]).T

        # Rotate sensor axes to global frame
        x_rot = R_current @ x_axis
        y_rot = R_current @ y_axis
        z_rot = R_current @ z_axis

        # Plot rotated sensor axes
        ax.quiver(*origin, *x_rot, color='r', label='X-axis (Red)')
        ax.quiver(*origin, *y_rot, color='g', label='Y-axis (Green)')
        ax.quiver(*origin, *z_rot, color='b', label='Z-axis (Blue)')

        # Plot the acceleration vector (from the origin)
        # Optionally, scale the acceleration vector for better visualization.
        scale = 0.5  # adjust scaling as needed
        ax.quiver(0, 0, 0, acc[0]*scale, acc[1]*scale, acc[2]*scale,
                  color='k', label='Acceleration (Black)', arrow_length_ratio=0.1)

        # Set plot formatting
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.set_zlim([-10, 10])
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        ax.set_title(f"IMU Orientation and Acceleration (Frame {frame})")
        ax.legend()

    # Create the animation and keep a reference to prevent garbage collection.
    ani = animation.FuncAnimation(fig, update, frames=len(rotation_matrices), 
                                  interval=1000/freq)
    plt.show()
