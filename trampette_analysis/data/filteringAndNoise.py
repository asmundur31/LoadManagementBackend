import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import imufusion
import matplotlib.pyplot as plt

def estimate_gyro_bias(sensor_df, gyro_cols, phases=None):
    df = sensor_df.copy()
    if phases is not None:
        df = df[df['label'].isin(phases)]
    gyro_bias = df[gyro_cols].mean(axis=0).values
    return gyro_bias

def fit_magnetometer_ellipsoid(mag_data):
    x, y, z = mag_data[:, 0], mag_data[:, 1], mag_data[:, 2]
    A = np.c_[2*x, 2*y, 2*z, np.ones(len(x))]
    b = x**2 + y**2 + z**2
    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    bias = coeffs[:3]
    return bias

def estimate_acc_bias(sensor_df, acc_cols, phases=None):
    df = sensor_df.copy()
    if phases is not None:
        df = df[df['label'].isin(phases)]
    gravity_vector = np.array([0, 0, -9.81])
    acc_bias = df[acc_cols].mean(axis=0).values - gravity_vector
    return acc_bias


def correct_sensor_bias(sensor_df, bias, sensor_cols):
    corrected_df = sensor_df.copy()
    corrected_df[sensor_cols] = corrected_df[sensor_cols] - bias
    return corrected_df


def get_orientation(df, acc_cols, gyr_cols, mag_cols, fs=104.0):
    timestamp = df['timestamp'].values
    acc_data = df[acc_cols].values
    gyr_data = df[gyr_cols].values
    mag_data = df[mag_cols].values
    delta_time = np.diff(timestamp, prepend=timestamp[0])

    ahrs = imufusion.Ahrs()
    ahrs.settings = imufusion.Settings(
        imufusion.CONVENTION_NED,  # convention
        0.5,  # gain
        2000,  # gyroscope range
        10,  # acceleration rejection
        10,  # magnetic rejection
        int(5 * fs),  # recovery trigger period = 5 seconds
    )
    q = np.empty((len(acc_data), 4))
    for index in range(len(df[acc_cols])):
        ahrs.update(gyr_data[index], acc_data[index], mag_data[index], delta_time[index])
        q[index] = ahrs.quaternion.wxyz
    return q

def rotate_data_into_earth_frame(df, quaternions, acc_cols, gyr_cols, mag_cols):
    """
    Rotates accelerometer, gyroscope, and magnetometer data into the Earth frame using quaternions.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        quaternions (np.ndarray): Nx4 array of [w, x, y, z] quaternions for each row.
        acc_cols (list): Names of columns containing accelerometer data.
        gyr_cols (list): Names of columns containing gyroscope data.
        mag_cols (list): Names of columns containing magnetometer data.

    Returns:
        pd.DataFrame: Updated DataFrame with rotated sensor data.
    """
    df_earth = df.copy()
    for i in range(len(df)):
        R = q2R(quaternions[i])  # Convert quaternion to rotation matrix
        df_earth.loc[i, acc_cols] = R @ df_earth.loc[i, acc_cols].values
        df_earth.loc[i, gyr_cols] = R @ df_earth.loc[i, gyr_cols].values
        df_earth.loc[i, mag_cols] = R @ df_earth.loc[i, mag_cols].values

    return df_earth

def q2R(q):
    """
    Converts a quaternion into a rotation matrix, with normalization.

    Parameters:
        q (np.array): Quaternion [w, x, y, z]

    Returns:
        np.array: 3x3 rotation matrix
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
    ])


def remove_gravity(df: pd.DataFrame, acc_col, g: float = 9.81) -> pd.DataFrame:
    """
    Removes gravity from accelerometer data.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        acc_cols (list): Names of columns containing accelerometer data.
        g (float): Gravitational acceleration (default is 9.81 m/s^2).

    Returns:
        pd.DataFrame: Updated DataFrame with gravity removed from accelerometer data.
    """
    df_gravity_removed = df.copy()
    print(f"Removing gravity from {acc_col}")
    df_gravity_removed[acc_col] += g
    return df_gravity_removed


def butterworth_filter_low(df, columns, cutoff=85, fs=104.0, order=4):
    filtered_data = df.copy()
    b, a = butter(order, cutoff, btype='low', analog=False, fs=fs)
    for col in columns:
        filtered_data[col] = filtfilt(b, a, filtered_data[col])
    return filtered_data

def butterworth_filter_high(df, columns, cutoff=85, fs=104.0, order=4):
    filtered_data = df.copy()
    b, a = butter(order, cutoff, btype='high', analog=False, fs=fs)
    for col in columns:
        print(col)
        filtered_data[col] = filtfilt(b, a, filtered_data[col])
    return filtered_data

def downsample(df, factor):
    """
    Downsamples the dataframe by selecting every Nth row.

    Parameters:
        df (pd.DataFrame): Input DataFrame with uniformly sampled data.
        factor (int): Downsampling factor (e.g., 2 keeps every 2nd row).

    Returns:
        pd.DataFrame: Downsampled DataFrame.
    """
    if factor < 1:
        raise ValueError("Downsampling factor must be >= 1")
    return df.iloc[::factor].reset_index(drop=True)


import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
def perform_fft(signal, title, fs=104.0):
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1 / fs)[:N//2]

    plt.figure(figsize=(10, 4))
    plt.plot(xf, 2.0/N * np.abs(yf[:N//2]))
    plt.title(f"FFT of IMU signal: {title}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()