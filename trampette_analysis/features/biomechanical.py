import pandas as pd
import numpy as np
from data.loader import get_subject_info
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d


def velocity(sensor_df: pd.DataFrame, raw_sensor_df: pd.DataFrame, placement: str, use_zupt: bool = False) -> pd.DataFrame:
    df = sensor_df.copy()
    raw_df = raw_sensor_df.copy()
    acc_x = df[f'{placement}_accx'].to_numpy()
    acc_y = df[f'{placement}_accy'].to_numpy()
    acc_z = df[f'{placement}_accz'].to_numpy()
    gyro_z = raw_df[f'{placement}_gyroz'].to_numpy()
    # Create a mask for the stance phase
    stance_mask = np.ones(len(gyro_z), dtype=bool)
    if use_zupt:
        # Lets start by creating a stance phase mask
        threshold = 100 # Gyroscope threshold for peak detection
        window_size = 20 # window size for peak detection
        peaks, _ = find_peaks(gyro_z, height=threshold, distance=window_size)
        for peak_idx in peaks:
            start = max(0, peak_idx - window_size)
            end = min(len(gyro_z), peak_idx + window_size)
            pre_valley_idx = start + np.argmin(gyro_z[start:peak_idx])
            post_valley_idx = peak_idx + np.argmin(gyro_z[peak_idx:end])
            # Mark stance phase as False between the two valleys (i.e., swing phase)
            stance_mask[pre_valley_idx:post_valley_idx+1] = False

    df['mask'] = stance_mask

    # Time difference
    dt = np.diff(df['timestamp'].to_numpy(), prepend=df['timestamp'].iloc[0])

    velocity = np.zeros((len(acc_x), 3))
    for i in range(1, len(acc_x)):
        velocity[i,0] = velocity[i-1,0] + 0.5 * (acc_x[i] + acc_x[i-1]) * dt[i]
        velocity[i,1] = velocity[i-1,1] + 0.5 * (acc_y[i] + acc_y[i-1]) * dt[i]
        velocity[i,2] = velocity[i-1,2] + 0.5 * (acc_z[i] + acc_z[i-1]) * dt[i]

        if use_zupt and stance_mask[i]:
            velocity[i, :] = 0
    
    df[f'{placement}_velocity_x'] = velocity[:, 0]
    df[f'{placement}_velocity_y'] = velocity[:, 1]
    df[f'{placement}_velocity_z'] = velocity[:, 2]
    return df
   
def jerk(sensor_df: pd.DataFrame, placement: str, fs: float) -> pd.DataFrame:
    df = sensor_df.copy()
    acc = df[[f'{placement}_accx', f'{placement}_accy', f'{placement}_accz']].to_numpy()
    jerk = np.gradient(acc, axis=0) * fs
    jerk_magnitude = np.linalg.norm(jerk, axis=1)
    df[f"{placement}_jerk_magnitude"] = jerk_magnitude
    return df

def impulse(sensor_df: pd.DataFrame, subject_id: int, placement: str, fs: float) -> pd.DataFrame:
    df = sensor_df.copy()
    subject_info = get_subject_info(subject_id)
    mass = subject_info['Weight (kg)'][subject_id - 7]
    dt = 1 / fs
    vertical_acc = df[f'{placement}_accz'].to_numpy()
    vertical_impulse = np.cumsum(vertical_acc) * dt * mass
    df[f"{placement}_vertical_impulse"] = vertical_impulse
    return df

def detect_takeoff_and_landing(jump_imu_df_earth, placement: str):
    acc = jump_imu_df_earth[f"{placement}_acc_magnitude"].to_numpy()

    Tm = 40
    t1 = np.argmax(acc)
    # Define window bounds
    start = int(max(t1 - Tm//2, 0))
    end = int(min(t1 + Tm//2, len(acc)))
    mask = np.ones(len(acc), dtype=bool)
    mask[start:end] = False

    t2 = np.argmax(acc[mask])
    t2 = np.arange(len(acc))[mask][t2]
    t_takeoff = min(t1, t2)
    t_landing = max(t1, t2)

    T_max_takeoff = 20
    T_max_landing = 40

    t_takeoff = t_takeoff + np.argmin(acc[t_takeoff:t_takeoff+T_max_takeoff//2])
    t_landing = t_landing-T_max_landing//2 + np.argmin(acc[t_landing-T_max_landing//2:t_landing])

    return t_takeoff, t_landing

def jump_height(ToF: float, subject_id: int) -> float:
    subject_info = get_subject_info(subject_id)
    s_height = subject_info['Height (m)'][subject_id-7]
    hCOM = s_height*0.5738 # 57.38% of gymnast height
    dhCOM = 0.307 # Average height difference between landing and trampette
    g = 9.81
    height = hCOM+(g/(8*ToF**2))*(ToF**2+2*dhCOM/g)**2
    return height

def energy(fss: float, jump_height: float, mass: float, position: str, flight_df: pd.DataFrame):
    # Compute the kinetic and potential energy
    kinetic_energy = 0.5*mass*fss**2
    potential_energy = mass*9.81*jump_height
    # Get the top range of the flight phase
    i = len(flight_df)//2
    window_size = 10
    half_window = window_size // 2
    start_idx = max(i - half_window, 0)
    end_idx = min(i + half_window, len(flight_df))
    # Extract the angular velocity data
    omega_x = flight_df["lower_back_gyrox"].iloc[start_idx:end_idx].to_numpy()
    omega_y = flight_df["lower_back_gyroy"].iloc[start_idx:end_idx].to_numpy()
    omega_z = flight_df["lower_back_gyroz"].iloc[start_idx:end_idx].to_numpy()
    # Smooth the angular velocity data
    omega_x = uniform_filter1d(omega_x, size=5)
    omega_y = uniform_filter1d(omega_y, size=5)
    omega_z = uniform_filter1d(omega_z, size=5)
    # Convert from degrees/s to radians/s
    omega_x_rad = np.deg2rad(omega_x)
    omega_y_rad = np.deg2rad(omega_y)
    omega_z_rad = np.deg2rad(omega_z)
    # Get the moment of inertia values
    Ix, Iy, Iz = get_inertia(position)
    # Compute the rotational energy
    rotational_energy = 0.5*Ix*omega_x_rad**2 + 0.5*Iy*omega_y_rad**2 + 0.5*Iz*omega_z_rad**2
    # Average the rotational energy over the top of the flight phase
    re_avg = np.mean(rotational_energy)
    return kinetic_energy, potential_energy, re_avg

def get_inertia(position: str):
    convertion = 0.112984829
    if position == "Straight":
        Ix = 115.0
        Iy = 103.0
        Iz = 11.3
    elif position == "Pike":
        Ix = 62.4
        Iy = 68.1
        Iz = 33.8
    elif position == "Tuck":
        Ix = 39.1
        Iy = 38.0
        Iz = 26.3
    else:
        raise ValueError("Invalid position. Choose from 'Tuck', 'Pike', or 'Straight'.")
    return Ix*convertion, Iy*convertion, Iz*convertion