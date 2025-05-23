import json
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from GraphPlotter import GraphPlotter
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt

def cumtrapz(y: np.ndarray, x: np.ndarray, axis: int = 0, initial: float = 0.0) -> np.ndarray:
    """
    Compute the cumulative integral of y with respect to x using the trapezoidal rule.
    
    Parameters:
      y (np.ndarray): Array of dependent values to integrate.
      x (np.ndarray): Array of independent values. Must be the same shape as y along the integration axis,
                      or 1D if it represents the sample points along that axis.
      axis (int): Axis along which to integrate (default is 0).
      initial (float): Initial value for the integration (default is 0.0).
      
    Returns:
      np.ndarray: An array of the same shape as y with the cumulative integral computed along the specified axis.
    """
    y = np.asarray(y)
    x = np.asarray(x)
    
    # Ensure x is broadcastable to y along the integration axis.
    # If x is 1D, reshape it for proper broadcasting.
    if x.ndim == 1:
        # Create an indexer that places the 1D x along the given axis.
        new_shape = [1] * y.ndim
        new_shape[axis] = x.size
        x = x.reshape(new_shape)
    
    # Compute the differences along the integration axis.
    dx = np.diff(x, axis=axis)
    
    # Compute the average of y values over adjacent intervals.
    # This averages y[i] and y[i+1] along the integration axis.
    slice1 = [slice(None)] * y.ndim
    slice2 = [slice(None)] * y.ndim
    slice1[axis] = slice(0, -1)
    slice2[axis] = slice(1, None)
    y_avg = 0.5 * (y[tuple(slice1)] + y[tuple(slice2)])
    
    # Compute the incremental areas (trapezoids)
    increments = y_avg * dx
    
    # Perform cumulative summation along the integration axis.
    cum_int = np.cumsum(increments, axis=axis)
    
    # Prepend the initial value along the integration axis.
    # First, determine the shape for the initial slice.
    init_shape = list(cum_int.shape)
    init_shape[axis] = 1
    initial_array = np.full(init_shape, initial, dtype=cum_int.dtype)
    
    # Concatenate the initial value with the cumulative integration result.
    result = np.concatenate([initial_array, cum_int], axis=axis)
    
    return result

def rotation_matrix_from_two_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute a rotation matrix that rotates vector a onto vector b.
    Both a and b are assumed to be 3D and normalized.
    """
    v = np.cross(a, b)
    c = np.dot(a, b)
    if np.linalg.norm(v) < 1e-8:
        # Vectors are nearly aligned
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    R = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (np.linalg.norm(v) ** 2))
    return R

def update_rotation(R: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """
    Update rotation matrix R given angular velocity omega (in rad/s) over time dt.
    Uses a simple Euler integration. For more accuracy, use more sophisticated integration.
    """
    omega = np.deg2rad(omega)
    # Compute the skew-symmetric matrix of omega
    omega_skew = np.array([[0, -omega[2], omega[1]],
                           [omega[2], 0, -omega[0]],
                           [-omega[1], omega[0], 0]])
    dR = omega_skew @ R * dt
    return R + dR  # For small dt, this approximates the new rotation

class Sensor:
    def __init__(self):
        """
        Initializes an empty IMU sensor object.
        """
        self.recording_info: Dict[str, Any] = {}
        self.recording_data: pd.DataFrame = pd.DataFrame()

    def load_json(self, file_path: str) -> None:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Process recording_info
        self.recording_info = data.get('recording_info', {})
        print("Recording Info:")
        for key, value in self.recording_info.items():
            if isinstance(value, list):
                print(f"  {key}: {len(value)} items")
            else:
                print(f"  {key}: {value}")
        
        # Process recording_data (expected to be a dict of lists)
        rec_data = data.get('recording_data', {})
        if isinstance(rec_data, dict):
            # Exclude 'average' and 'rrdata' keys
            keys_to_exclude = ['average', 'rrdata']
            for key in keys_to_exclude:
                if key in rec_data:
                    rec_data.pop(key)
            
            # Ensure all lists have the same length as the 'timestamp' list.
            timestamps = rec_data.get('timestamp')
            if timestamps is None or not isinstance(timestamps, list):
                print("Error: 'timestamp' key is missing or not a list in recording_data.")
                self.recording_data = pd.DataFrame(rec_data)
                return
            
            desired_length = len(timestamps)
            for key, value in rec_data.items():
                if isinstance(value, list):
                    if len(value) > desired_length:
                        rec_data[key] = value[:desired_length]
                    elif len(value) < desired_length:
                        print(f"Warning: List for '{key}' is shorter than 'timestamp' list. It has {len(value)} items; expected {desired_length}.")
            
            print("\nRecording Data (after trimming and exclusion):")
            for key, value in rec_data.items():
                if isinstance(value, list):
                    print(f"  {key}: {len(value)} items")
            
            # Create a DataFrame from the dict.
            self.recording_data = pd.DataFrame(rec_data)
        else:
            print("recording_data is not a dictionary of lists.")
            self.recording_data = pd.DataFrame()

    def add_json_data(self, json_data: Dict[str, Any]) -> None:
        """
        Adds additional IMU data from a JSON structure.
        Expects recording_data to be an object with lists.
        Prints the length of each list, trims lists to the timestamp length,
        and excludes the 'average' and 'rrdata' keys before appending.
        
        :param json_data: Dictionary representing the JSON data.
        """
        # Merge or update recording_info if needed
        if not self.recording_info:
            self.recording_info = json_data.get('recording_info', {})
        else:
            self.recording_info.update(json_data.get('recording_info', {}))
        
        new_rec_data = json_data.get('recording_data', {})
        if isinstance(new_rec_data, dict):
            print("\nAdditional Recording Data (before trimming and exclusion):")
            for key, value in new_rec_data.items():
                if isinstance(value, list):
                    print(f"  {key}: {len(value)} items")
                else:
                    print(f"  {key}: not a list")
            
            # Exclude 'average' and 'rrdata'
            keys_to_exclude = ['average', 'rrdata']
            for key in keys_to_exclude:
                if key in new_rec_data:
                    new_rec_data.pop(key)
            
            # Ensure lists match the length of the timestamp list
            timestamps = new_rec_data.get('timestamp')
            if timestamps is None or not isinstance(timestamps, list):
                print("Error: 'timestamp' key is missing or not a list in additional recording_data.")
            else:
                desired_length = len(timestamps)
                for key, value in new_rec_data.items():
                    if isinstance(value, list):
                        if len(value) > desired_length:
                            new_rec_data[key] = value[:desired_length]
                        elif len(value) < desired_length:
                            print(f"Warning: Additional list for '{key}' is shorter than 'timestamp' list.")
            
            print("\nAdditional Recording Data (after trimming and exclusion):")
            for key, value in new_rec_data.items():
                if isinstance(value, list):
                    print(f"  {key}: {len(value)} items")
            
            new_data_df = pd.DataFrame(new_rec_data)
            self.recording_data = pd.concat([self.recording_data, new_data_df], ignore_index=True)
        else:
            print("Additional recording_data is not a dictionary of lists.")

    def _get_data_segment(self, start_time: float, end_time: float) -> pd.DataFrame:
        """
        Retrieves a segment of the recording data between start_time and end_time.
        Assumes that the 'timestamp' column exists and is numeric.
        
        :param start_time: Start time of the segment.
        :param end_time: End time of the segment.
        :return: DataFrame slice of the data within the time interval.
        """
        segment = self.recording_data[
            (self.recording_data['timestamp'] >= start_time) & 
            (self.recording_data['timestamp'] <= end_time)
        ].reset_index(drop=True)
        return segment

    def calculate_linear_sprint_speed(self, start_time: float, end_time: float, static_time: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the linear velocity (in the global frame) and 3D displacement for each timestamp in the segment.
        
        Assumptions/Steps:
          1. During the first `static_time` seconds of the segment, the sensor is static.
          2. The static phase is used to determine:
             - The sensor’s orientation relative to the global frame.
             - The sensor bias for acceleration, gyroscope, and magnetometer.
             - Gravity removal for accelerometer bias.
          3. For the dynamic phase:
             - The sensor bias is removed from the raw data.
             - The sensor’s orientation is estimated by integrating the bias‐corrected gyroscope data (starting from the static orientation).
             - The dynamic accelerations are rotated to the global frame.
             - Gravity is removed from the global-frame acceleration.
             - Integration (with simple filtering/integration) is used to obtain velocity and displacement.
        
        Returns:
          - velocities: A (N x 3) array of velocity (m/s) for each timestamp.
          - displacements: A (N x 3) array of displacement (m) for each timestamp.
        """
        # Extract the segment data
        segment = self._get_data_segment(start_time, end_time)
        times = segment['timestamp'].to_numpy()
        # Ensure times are in seconds
        dt = np.mean(np.diff(times))
        
        # Extract sensor measurements as numpy arrays
        acc = segment[['accx', 'accy', 'accz']].to_numpy()
        gyro = segment[['gyrox', 'gyroy', 'gyroz']].to_numpy()
        mag = segment[['magnx', 'magny', 'magnz']].to_numpy()
        
        # Identify static and dynamic phases
        static_end_time = start_time + static_time
        static_mask = times <= static_end_time
        dynamic_mask = times > static_end_time
        if np.sum(static_mask) < 2:
            raise ValueError("Not enough static samples for calibration.")
        
        # 1. Compute static orientation.
        # Average static accelerometer reading (includes gravity)
        acc_static_mean = np.mean(acc[static_mask], axis=0)
        # Normalize to get sensor-frame gravity direction.
        acc_static_norm = acc_static_mean / np.linalg.norm(acc_static_mean)
        # Assume global gravity is [0, 0, -1] (direction only)
        gravity_global_dir = np.array([0, -1, 0])
        # Compute rotation from sensor frame to global frame during static phase.
        R_static = rotation_matrix_from_two_vectors(acc_static_norm, gravity_global_dir)
        
        # 2. Compute sensor biases during static phase.
        bias_acc = np.mean(acc[static_mask], axis=0) - np.linalg.inv(R_static) @ (gravity_global_dir * 9.81)
        bias_gyro = np.mean(gyro[static_mask], axis=0)
        bias_mag = np.mean(mag[static_mask], axis=0)
        
        # For debugging:
        print("Static calibration:")
        print("  R_static:\n", R_static)
        print("  bias_acc:", bias_acc)
        print("  bias_gyro:", bias_gyro)
        print("  bias_mag:", bias_mag)
        
        # 3. Process dynamic phase:
        # Remove sensor bias from dynamic measurements.
        acc_dynamic = acc.copy()
        gyro_dynamic = gyro.copy()
        # Remove bias from all samples (static and dynamic), although static phase should be near zero.
        acc_dynamic = acc_dynamic - bias_acc
        gyro_dynamic = gyro_dynamic - bias_gyro
        
        # Initialize array for dynamic orientation matrices.
        N = len(times)
        R_dynamic = [None] * N
        # For t=0 (start of the segment), use the static orientation.
        R_current = R_static.copy()
        R_dynamic[0] = R_current
        
        # For each subsequent sample, update the orientation using the (bias-corrected) gyro.
        for i in range(1, N):
            dt_i = times[i] - times[i-1]
            # Update the rotation matrix with the angular velocity vector at the previous timestep.
            R_current = update_rotation(R_current, gyro_dynamic[i-1], dt_i)
            # Re-orthogonalize if desired (here we assume small dt makes drift minimal)
            R_dynamic[i] = R_current
        
        # Convert list of rotation matrices to a numpy array for easier indexing.
        R_dynamic = np.array(R_dynamic)  # Shape (N, 3, 3)
        
        # 4. Transform acceleration measurements from sensor frame to global frame.
        acc_global = np.empty_like(acc_dynamic)
        for i in range(N):
            # Rotate acceleration to global frame:
            acc_global[i] = R_dynamic[i] @ acc_dynamic[i]
        
        # 5. Remove gravity from global acceleration.
        gravity_global = np.array([0, -9.81, 0])
        acc_global_corrected = acc_global - gravity_global

        # 6. Integrate acceleration to get velocity.
        # We integrate over the entire segment. We use cumulative trapezoidal integration.
        # For the static phase we expect near zero velocity.
        velocities = np.zeros_like(acc_global_corrected)
        # cumtrapz returns an array of length N-1; we pad the first sample with zero.
        v_dynamic = cumtrapz(acc_global_corrected, times, axis=0, initial=0)
        velocities = v_dynamic
        
        # 7. Integrate velocity to get displacement.
        displacements = np.zeros_like(velocities)
        d_dynamic = cumtrapz(velocities, times, axis=0, initial=0)
        displacements = d_dynamic
        
        # Optionally, you can apply filtering/smoothing to velocity and displacement.
        
        # Return velocity and displacement for each timestamp in the segment.
        return velocities, displacements

    def calculate_jerk(self, start_time: float, end_time: float) -> float:
        """
        Calculates the magnitude of the jerk (i.e. the time derivative of acceleration)
        in the sensor frame. It assumes that the acceleration data (accX, accY, accZ)
        in the recording_data has already been corrected for gravity and sensor bias.
        
        Steps:
          1. Extract the segment of data between start_time and end_time.
          2. Retrieve timestamps and the gravity-corrected acceleration (in sensor frame).
          3. Compute the numerical derivative (jerk vector) using finite differences.
             - Use central differences for interior points.
             - Use forward/backward differences at the endpoints.
          4. Compute the magnitude (norm) of the jerk vector for each timestamp.
          5. Apply a simple moving average filter to the jerk magnitude to reduce noise.
        
        Returns:
          A numpy array containing the filtered jerk magnitude for each timestamp in the segment.
        """
        # 1. Extract the segment.
        segment = self._get_data_segment(start_time, end_time)
        times = segment['timestamp'].to_numpy()  # assuming these are in seconds
        acc = segment[['accx', 'accy', 'accz']].to_numpy()  # gravity-corrected acceleration
        # Plot the raw acceleration data
        plotter = GraphPlotter(AccelerationX=acc[:, 0], AccelerationY=acc[:, 1], AccelerationZ=acc[:, 2])
        plotter.plot2d(title="Raw Acceleration Data", xlabel="Sample Index", ylabel="Acceleration (m/s²)")
        N = len(times)
        if N < 2:
            raise ValueError("Not enough data points to compute jerk.")
        
        # 2. Pre-allocate an array for the jerk vectors (3 components per timestamp)
        jerk_vectors = np.zeros((N, 3))
        
        # 3. Compute finite differences.
        # For the first sample, use forward difference.
        dt0 = times[1] - times[0]
        jerk_vectors[0] = (acc[1] - acc[0]) / dt0
        
        # For interior samples, use central differences.
        for i in range(1, N - 1):
            dt = times[i + 1] - times[i - 1]
            if dt == 0:
                jerk_vectors[i] = np.zeros(3)
            else:
                jerk_vectors[i] = (acc[i + 1] - acc[i - 1]) / dt
        
        # For the last sample, use backward difference.
        dt_end = times[-1] - times[-2]
        jerk_vectors[-1] = (acc[-1] - acc[-2]) / dt_end
        
        # 4. Compute the magnitude of the jerk vector at each time step.
        jerk_magnitude = np.linalg.norm(jerk_vectors, axis=1)
        
        # 5. Apply a simple moving average filter to smooth the jerk magnitude.
        #    (You can change the window size or filtering method later.)
        window_size = 5  # for example, a window of 5 samples
        filtered_jerk = self._moving_average(jerk_magnitude, window_size)
        
        return filtered_jerk

    def calculate_impulse(self, 
                          start_time: float, 
                          end_time: float,
                          mass: float,
                          static_time: float = 2.0,
                          filter_window: int = 5) -> np.ndarray:
        """
        Calculates the instantaneous impulse (in N·s) for a given time interval.
        
        The impulse is computed as the time integral of the (gravity‐ and bias‐corrected)
        acceleration multiplied by mass. The steps are as follows:
        
          1. Extract the data segment between start_time and end_time.
          2. Use the first static_time seconds to estimate sensor bias (including gravity)
             on each acceleration axis (accX, accY, accZ). This bias is subtracted so that
             at rest the acceleration is near zero.
          3. Apply a moving average filter to smooth the corrected acceleration.
          4. Numerically integrate the filtered acceleration over time using a cumulative
             trapezoidal rule. This yields the change in velocity.
          5. Multiply the change in velocity by the mass to obtain the impulse (which equals
             the change in momentum).
        
        Parameters:
          start_time (float): Start time of the segment (s).
          end_time (float): End time of the segment (s).
          mass (float): Subject mass (kg).
          static_time (float): Duration (s) used for static calibration (default is 2.0 s).
          filter_window (int): Window size for moving average filtering.
        
        Returns:
          times (np.ndarray): Time stamps corresponding to the impulse calculation.
          impulse (np.ndarray): The instantaneous impulse vector (N·s) for each timestamp.
                              This is a (N x 3) array corresponding to [impulse_x, impulse_y, impulse_z].
        """
        # 1. Extract the data segment.
        segment = self._get_data_segment(start_time, end_time)
        if segment.empty:
            raise ValueError("No data in the specified time range.")
        times = segment['timestamp'].to_numpy()  # assumed in seconds
        # Extract acceleration data (assumed to be in m/s²)
        acc_data = segment[['accx', 'accy', 'accz']].to_numpy()
        
        # 2. Static calibration: use the first static_time seconds for bias/gravity estimation.
        static_end = start_time + static_time
        static_mask = (segment['timestamp'] >= start_time) & (segment['timestamp'] <= static_end)
        if not any(static_mask):
            raise ValueError("No static samples available for calibration.")
        bias = np.mean(acc_data[static_mask], axis=0)
        # Remove bias (and gravity) from all acceleration data.
        acc_corrected = acc_data - bias
        
        # 3. Filter the corrected acceleration for each axis.
        acc_filtered = np.empty_like(acc_corrected)
        for axis in range(3):
            acc_filtered[:, axis] = self._moving_average(acc_corrected[:, axis], filter_window)
        
        # 4. Compute the cumulative integral (change in velocity) for each axis.
        # We'll integrate along the time axis.
        # For each axis, use the helper _cumtrapz.
        delta_v = np.empty_like(acc_filtered)
        for axis in range(3):
            delta_v[:, axis] = cumtrapz(acc_filtered[:, axis], times)
        
        # 5. Calculate impulse vector: impulse = mass * delta_v.
        impulse = mass * delta_v
        
        # Optionally, plot the filtered acceleration and the integrated velocity.
        plt.figure(figsize=(10, 6))
        plt.subplot(2,1,1)
        plt.plot(times, acc_filtered)
        plt.xlabel('Time (s)')
        plt.ylabel('Filtered Acceleration (m/s²)')
        plt.title('Filtered, Bias-Corrected Acceleration')
        plt.legend(['accX','accY','accZ'])
        plt.grid(True)
        
        plt.subplot(2,1,2)
        plt.plot(times, delta_v)
        plt.xlabel('Time (s)')
        plt.ylabel('Delta Velocity (m/s)')
        plt.title('Integrated Velocity (Change in Velocity)')
        plt.legend(['ΔvX','ΔvY','ΔvZ'])
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        return impulse

    def _moving_average(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """
        Applies a simple moving average filter to 1D data.
        
        Parameters:
          data: 1D numpy array.
          window_size: Size of the moving window.
        
        Returns:
          Filtered data as a 1D numpy array of the same length as input.
        """
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        
        # Use 'valid' mode to avoid edge effects, then pad the edges to maintain length.
        cumsum = np.cumsum(np.insert(data, 0, 0))
        filtered = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
        
        # Pad the start and end to keep the same length.
        pad_front = np.full(window_size // 2, filtered[0])
        pad_end = np.full(data.shape[0] - filtered.shape[0] - pad_front.shape[0], filtered[-1])
        return np.concatenate((pad_front, filtered, pad_end))

    def _low_pass_filter(self, data, cutoff, fs, order=4):
        """Applies a low-pass Butterworth filter."""
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data)

    def calculate_potential_energy_at_max_height(self, mass: float, jump_info: dict, g: float = 9.81) -> float:
        """
        Calculates the potential energy at the maximal jump height.
        
        Parameters:
          mass (float): Mass of the subject (in kg).
          jump_info (dict): Dictionary containing jump detection information. 
                            Must include the key 'jump_height' (in meters).
          g (float): Gravitational acceleration (default is 9.81 m/s²).
        
        Returns:
          float: The potential energy (in Joules) calculated as:
                 PE = mass * g * jump_height.
        
        If jump_info is empty or does not contain 'jump_height', returns 0.
        """
        if not jump_info or 'jump_height' not in jump_info:
            print("No jump information provided or jump height not found.")
            return 0.0
        
        jump_height = jump_info['jump_height']
        potential_energy = mass * g * jump_height
        
        return potential_energy

    def calculate_kinetic_energy(self, start_time: float, end_time: float, mass: float, static_time: float = 2.0) -> np.ndarray:
        """
        Calculates the instantaneous kinetic energy over the given time segment.
        
        The kinetic energy is computed using the linear velocity obtained from the speed calculation:
            KE = 0.5 * mass * ||v||^2
        
        Parameters:
          start_time (float): Start time of the segment.
          end_time (float): End time of the segment.
          mass (float): Mass of the subject (in kg).
          static_time (float): Duration (in seconds) of the static phase for calibration (default is 2.0).
        
        Returns:
          np.ndarray: An array of instantaneous kinetic energy (in Joules) for each timestamp in the segment.
        """
        # 1. Get the velocity using the speed method.
        velocities, _ = self.calculate_linear_sprint_speed(start_time, end_time, static_time)
        # 2. Compute the squared magnitude of the velocity at each timestamp.
        #    This is computed as the sum of the squares of the x, y, and z components.
        v_squared = np.sum(velocities ** 2, axis=1)
        # 3. Compute the kinetic energy using KE = 0.5 * mass * v^2.
        kinetic_energy = 0.5 * mass * v_squared
        
        return kinetic_energy
    
    def calculate_rotational_energy(self, start_time: float, end_time: float, body_position: str, static_time: float = 2.0, filter_window: int = 5, mass: float = 70.0, height: float = 1.75) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the instantaneous rotational energy (J) during the airborne phase 
        (detected from a segment between segment_start_time and segment_end_time)
        based on 3D gyroscope data and the gymnast's body position.
        
        This method first calls detect_airborne_phase to obtain takeoff and landing times.
        Then, it performs the following steps:
          1. Gyroscope bias calibration (using the first static_time seconds).
          2. Extraction of the airborne gyroscope data (between detected takeoff and landing).
          3. Filtering of the bias-corrected gyro data.
          4. Unit conversion from degrees per second to radians per second.
          5. Computation of the angular velocity magnitude (ω).
          6. Estimation of the moment of inertia using mass, height, and body_position.
          7. Calculation of instantaneous rotational energy: E_rot = 0.5 * I * ω².
        
        Parameters:
          segment_start_time (float): Start time of the overall segment.
          segment_end_time (float): End time of the overall segment.
          body_position (str): Body position during the jump ("tuck", "pike", or "straight").
          static_time (float): Duration (s) for static calibration.
          filter_window (int): Window size for moving average filtering.
          mass (float): Subject mass (kg).
          height (float): Subject height (m).
        
        Returns:
          times_air (np.ndarray): Timestamps during the airborne phase.
          rotational_energy (np.ndarray): Instantaneous rotational energy (J) for each timestamp.
        """
        # First, detect the airborne phase (takeoff and landing) from the provided segment.
        airborne_phase = self._detect_airborne_phase(start_time, end_time,
                                                      static_time=static_time,
                                                      filter_window=filter_window,
                                                      min_peak_height=40.0,         # Adjust as needed
                                                      min_peak_prominence=10.0,       # Adjust as needed
                                                      g=9.81)
        if not airborne_phase:
            raise ValueError("Airborne phase could not be detected.")
        takeoff_time = airborne_phase['takeoff_time']
        landing_time = airborne_phase['landing_time']
        
        # --- Gyroscope Bias Calibration ---
        static_segment = self._get_data_segment(start_time, start_time+static_time)  # using t=0 as start of recording
        if static_segment.empty:
            raise ValueError("No static data available for gyroscope calibration.")
        gyro_static = static_segment[['gyrox', 'gyroy', 'gyroz']].to_numpy()
        gyro_bias = np.mean(gyro_static, axis=0)
        
        # --- Extract Airborne Gyroscope Data ---
        airborne_segment = self._get_data_segment(takeoff_time, landing_time)
        if airborne_segment.empty:
            raise ValueError("No gyroscope data in the airborne phase.")
        times_air = airborne_segment['timestamp'].to_numpy()
        gyro_air = airborne_segment[['gyrox', 'gyroy', 'gyroz']].to_numpy()
        gyro_air_corrected = gyro_air - gyro_bias
        
        # --- Filtering ---
        gyro_air_filtered = np.empty_like(gyro_air_corrected)
        for i in range(3):
            gyro_air_filtered[:, i] = self._moving_average(gyro_air_corrected[:, i], filter_window)
        
        # --- Unit Conversion: degrees/s to radians/s ---
        gyro_air_rad = np.deg2rad(gyro_air_filtered)
        
        # --- Compute Angular Velocity Magnitude ---
        omega = np.linalg.norm(gyro_air_rad, axis=1)
        
        # --- Estimate Moment of Inertia (based on body position) ---
        I = self.estimate_moment_of_inertia(mass, height, body_position)
        
        # --- Compute Instantaneous Rotational Energy ---
        rotational_energy = 0.5 * I * (omega ** 2)
        
        # Optionally, plot the results.
        plt.figure(figsize=(10, 4))
        plt.subplot(2, 1, 1)
        plt.plot(times_air, omega, label='Angular Velocity (rad/s)')
        plt.xlabel('Time (s)')
        plt.ylabel('ω (rad/s)')
        plt.title('Airborne Angular Velocity')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(times_air, rotational_energy, label='Rotational Energy (J)', color='magenta')
        plt.xlabel('Time (s)')
        plt.ylabel('Energy (J)')
        plt.title('Instantaneous Rotational Energy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        return times_air, rotational_energy

    def estimate_moment_of_inertia(self, mass: float, height: float, body_position: str) -> float:
        """
        Estimates the moment of inertia (I) of the human body using a simple scaling law:
        
            I = k * mass * height^2
        
        The scaling factor k depends on the body position:
          - 'tuck': lower moment of inertia (e.g., k ~ 0.06)
          - 'pike': moderate (e.g., k ~ 0.08)
          - 'straight': higher (e.g., k ~ 0.10)
        
        Parameters:
          mass (float): Mass of the subject (kg).
          height (float): Height of the subject (m).
          body_position (str): One of "tuck", "pike", or "straight".
        
        Returns:
          float: Estimated moment of inertia (kg·m²).
        """
        position_factors = {
            "tuck": 0.06,
            "pike": 0.08,
            "straight": 0.10
        }
        bp = body_position.lower()
        if bp not in position_factors:
            raise ValueError("Invalid body_position. Choose 'tuck', 'pike', or 'straight'.")
        k = position_factors[bp]
        return k * mass * (height ** 2)

    def _detect_airborne_phase(self, 
                              start_time: float, 
                              end_time: float,
                              static_time: float = 2.0,
                              filter_window: int = 5,
                              min_peak_height: float = 40.0,
                              min_peak_prominence: float = 10.0,
                              g: float = 9.81) -> dict:
        """
        Detects takeoff and landing times within a segment from start_time to end_time
        using a big-peak detection method on the vertical acceleration (accY).
        
        Steps:
          1. Extract the data segment and perform static calibration using the first static_time seconds
             to remove gravity and sensor bias from accY.
          2. Filter the corrected signal.
          3. Use SciPy's find_peaks with provided parameters to detect big peaks.
          4. Assume the first detected big peak is takeoff and the second is landing.
        
        Returns:
          dict with keys: 'takeoff_time' and 'landing_time'
          If not detected, returns an empty dict.
        """
        segment = self._get_data_segment(start_time, end_time)
        if segment.empty:
            print("No data in the specified time range.")
            return {}
        times = segment['timestamp'].to_numpy()
        acc_y = segment['accy'].to_numpy()
        
        # Static calibration: use the first static_time seconds
        static_end = start_time + static_time
        static_mask = (segment['timestamp'] >= start_time) & (segment['timestamp'] <= static_end)
        if not any(static_mask):
            print("No static samples found for calibration.")
            return {}
        mean_static = np.mean(acc_y[static_mask])
        acc_y_corrected = acc_y - mean_static
        
        # Filter the corrected signal
        acc_y_filtered = self._moving_average(acc_y_corrected, window_size=filter_window)
        
        # Detect peaks using find_peaks
        peaks, properties = find_peaks(acc_y_filtered, height=min_peak_height, prominence=min_peak_prominence)
        if len(peaks) < 2:
            print(f"Jump not detected: only {len(peaks)} big peak(s) found.")
            self._plot_peak_detection(times, acc_y_filtered, peaks, properties)
            return {}
        
        # Assume first peak is takeoff, second is landing.
        takeoff_idx = peaks[0]
        landing_idx = peaks[1]
        takeoff_time = times[takeoff_idx]
        landing_time = times[landing_idx]
        
        # Plot for debugging
        self._plot_peak_detection(times, acc_y_filtered, peaks, properties, takeoff_idx, landing_idx)
        
        return {'takeoff_time': takeoff_time, 'landing_time': landing_time}
    
    def calculate_jump_height(self, start_time: float, end_time: float, static_time: float = 2.0, filter_window: int = 5, min_peak_height: float = 40.0, min_peak_prominence: float = 10.0, g: float = 9.81) -> dict:
        """
        Detects a single jump event using a peak-detection method that focuses on the large peaks 
        (big jump-related peaks) in the vertical (y-axis) acceleration signal.

        Steps:
          1. Extract the data segment between start_time and end_time.
          2. Use the first static_time seconds to compute the mean y-acceleration (gravity + bias)
             and subtract it from the entire segment.
          3. Apply a moving average filter to smooth the corrected acceleration signal.
          4. Use SciPy’s find_peaks with a minimum height and prominence (min_peak_height and 
             min_peak_prominence) to detect the big peaks.
          5. Assume that the first detected peak corresponds to takeoff and the second to landing.
          6. Compute the flight time as the difference between the times of these peaks.
          7. Compute jump height using the time-of-flight formula: h = (g * T²) / 8.
          8. Plot the filtered signal with the detected peaks for debugging.
        
        Parameters:
          start_time (float): Start time of the segment.
          end_time (float): End time of the segment.
          static_time (float): Duration (in seconds) for static calibration at the start.
          filter_window (int): Window size for the moving average filter.
          min_peak_height (float): Minimum height (in m/s²) for a peak to be considered.
          min_peak_prominence (float): Minimum prominence required for a peak.
          g (float): Gravitational acceleration (default 9.81 m/s²).
        
        Returns:
          dict: Contains 'takeoff_time', 'landing_time', 'flight_time', and 'jump_height'.
                If the jump is not detected, returns an empty dict.
        """
        # 1. Extract the data segment.
        segment = self._get_data_segment(start_time, end_time)
        if segment.empty:
            print("No data in the specified time range.")
            return {}

        times = segment['timestamp'].to_numpy()  # assumed to be in seconds
        acc_x = segment['accx'].to_numpy()
        acc_y = segment['accy'].to_numpy()
        acc_z = segment['accz'].to_numpy()

        # 2. Static calibration: use the first static_time seconds to compute gravity + bias.
        static_end = start_time + static_time
        static_mask = (segment['timestamp'] >= start_time) & (segment['timestamp'] <= static_end)
        if not any(static_mask):
            print("No static samples found in the specified static_time interval.")
            return {}

        # Remove gravity by calculating the mean of the acceleration in the static period.
        mean_static_x = np.mean(acc_x[static_mask])
        mean_static_y = np.mean(acc_y[static_mask])
        mean_static_z = np.mean(acc_z[static_mask])

        # Remove gravity + bias from each axis.
        acc_x_corrected = acc_x - mean_static_x
        acc_y_corrected = acc_y - mean_static_y
        acc_z_corrected = acc_z - mean_static_z
        
        # 3. Compute the resultant 3D acceleration magnitude.
        acc_mag = np.sqrt(acc_x_corrected**2 + acc_y_corrected**2 + acc_z_corrected**2)

        # 4. Apply low-pass filter to remove noise from the signal.
        fs = 1 / (times[1] - times[0])  # Sampling frequency (inverse of the time step)
        acc_mag_filtered = self._low_pass_filter(acc_mag, cutoff=5, fs=fs)  # Increased cutoff frequency

        # 5. Differentiate the filtered signal to capture rate of change.
        derivative = np.gradient(acc_mag_filtered)

        # 6. Debug plot: Check if the derivative shows spikes.
        plt.figure()
        plt.plot(times, derivative, label="Derivative of Acceleration")
        plt.legend()
        plt.show()

        # 7. Custom peak and valley detection with height difference threshold
        # Detect the peaks first using the derivative.
        peaks, properties = find_peaks(derivative, height=min_peak_height, prominence=min_peak_prominence, distance=50)

        if len(peaks) < 2:
            print(f"Jump not detected: not enough peaks found (found peaks: {len(peaks)})")
            self._plot_peak_detection(times, acc_mag, peaks, properties)
            return {}

        # 8. Refine takeoff and landing peaks by checking valley height difference.
        takeoff_idx = None
        landing_idx = None

        print("Number of detected peaks:", len(peaks))

        for i in range(0, len(peaks)):
            # For each peak, find the corresponding valley (local minimum) that comes after it.
            peak_time = times[peaks[i]]
            peak_value = derivative[peaks[i]]
            
            # Find the valley after the peak within a specified time window (e.g., 200ms)
            valley_mask = (times > peak_time) & (times < peak_time + 0.2)  # 200ms window after peak
            valley_indices = np.where(valley_mask)[0]
            if len(valley_indices) == 0:
                continue  # No valley found

            valley_idx = valley_indices[np.argmin(derivative[valley_indices])]
            valley_value = derivative[valley_idx]
            print("Peak value:", peak_value, "Valley value:", valley_value)

            # Check the difference in height between the peak and the valley
            if peak_value - valley_value >= 20:  # Minimum height difference threshold (50)
                if takeoff_idx is None:
                    takeoff_idx = peaks[i]  # First peak is takeoff
                elif landing_idx is None:
                    landing_idx = peaks[i]  # Second peak is landing
                    break  # Exit loop once both are found

        if takeoff_idx is None or landing_idx is None:
            print("Unable to detect both takeoff and landing peaks with sufficient height difference.")
            self._plot_peak_detection(times, acc_mag, peaks, properties, takeoff_idx, landing_idx)
            return {}

        # Temporal constraints: Ensure the landing is sufficiently after takeoff
        if times[landing_idx] - times[takeoff_idx] < 0.1:  # e.g., minimum 100ms between takeoff and landing
            print("Invalid jump: takeoff and landing detected too close together.")
            self._plot_peak_detection(times, acc_mag, peaks, properties, takeoff_idx, landing_idx)
            return {}

        takeoff_time = times[takeoff_idx]
        landing_time = times[landing_idx]
        flight_time = landing_time - takeoff_time

        # 9. Estimate jump height using flight time
        jump_height = (g * flight_time**2) / 8.0  # This is based on a simplified projectile model

        # 10. Plot the filtered signal with detected peaks
        self._plot_peak_detection(times, acc_mag_filtered, peaks, properties, takeoff_idx, landing_idx)

        # 11. Return the results
        return {
            'takeoff_time': takeoff_time,
            'landing_time': landing_time,
            'flight_time': flight_time,
            'jump_height': jump_height
        }
    
    def _plot_peak_detection(self, times, acc_y_filtered, peaks, properties, takeoff_idx=None, landing_idx=None):
        """
        Plots the filtered vertical acceleration signal with detected peaks marked.
        If takeoff_idx and landing_idx are provided, marks them in the plot.
        """
        plt.figure(figsize=(10, 4))
        plt.plot(times, acc_y_filtered, label='Filtered accY (corrected)', color='blue')
        plt.plot(times[peaks], acc_y_filtered[peaks], "x", color='red', label='Detected Peaks')
        # Optionally plot the peak heights as returned by find_peaks:
        plt.vlines(x=times[peaks], ymin=properties["prominences"], ymax=acc_y_filtered[peaks], color='gray', linestyle='--', label='Prominences')
        if takeoff_idx is not None:
            plt.axvline(x=times[takeoff_idx], color='green', linestyle='--', label='Takeoff')
        if landing_idx is not None:
            plt.axvline(x=times[landing_idx], color='orange', linestyle='--', label='Landing')
        plt.xlabel('Time (s)')
        plt.ylabel('Vertical Acc (m/s²) (corrected)')
        plt.title('Jump Detection via Big Peak Detection')
        plt.legend()
        plt.grid(True)
        plt.show()

    def detect_jumps(self, 
                     start_time: float, 
                     end_time: float,
                     static_time: float = 2.0,
                     filter_window: int = 5,
                     min_peak_height: float = 40.0,
                     min_peak_prominence: float = 10.0,
                     g: float = 9.81,
                     min_flight_time: float = 0.3,
                     max_flight_time: float = 2.0) -> list:
        """
        Detects jump events in a full recording segment that may contain multiple phases (static,
        sprint, jump, etc.) and returns a list of detected jumps. This method uses a peak-detection
        algorithm similar to the one used for jump height but adapted for potentially multiple events.
        
        Steps:
          1. Extract the full data segment between start_time and end_time.
          2. Perform static calibration using the first static_time seconds to remove gravity and sensor bias from accY.
          3. Filter the corrected vertical acceleration signal (accY) using a moving average.
          4. Use SciPy's find_peaks with specified min_peak_height and min_peak_prominence to detect candidate peaks.
          5. Scan through the candidate peaks in time order and pair peaks as takeoff and landing events:
              - A valid jump is defined as a pair of consecutive peaks with a flight time between min_flight_time and max_flight_time.
          6. For each valid jump, compute flight time and jump height: h = (g * T²) / 8.
        
        Parameters:
          start_time (float): Start time of the overall segment.
          end_time (float): End time of the overall segment.
          static_time (float): Duration (s) for static calibration at the start.
          filter_window (int): Window size for moving average filtering.
          min_peak_height (float): Minimum height for candidate peaks.
          min_peak_prominence (float): Minimum prominence for candidate peaks.
          g (float): Gravitational acceleration (default 9.81 m/s²).
          min_flight_time (float): Minimum flight time (s) to consider a valid jump.
          max_flight_time (float): Maximum flight time (s) to consider a valid jump.
        
        Returns:
          list of dictionaries, each with keys:
              'takeoff_time', 'landing_time', 'flight_time', 'jump_height'
          If no valid jump is detected, returns an empty list.
        """
        segment = self._get_data_segment(start_time, end_time)
        if segment.empty:
            print("No data in the specified time range.")
            return []
        times = segment['timestamp'].to_numpy()
        acc_y = segment['accy'].to_numpy()
        
        # 1. Static calibration: use first static_time seconds to compute gravity+bias.
        static_end = start_time + static_time
        static_mask = (segment['timestamp'] >= start_time) & (segment['timestamp'] <= static_end)
        if not any(static_mask):
            print("No static samples available for calibration.")
            return []
        mean_static = np.mean(acc_y[static_mask])
        acc_y_corrected = acc_y - mean_static
        
        # 2. Filter the corrected signal.
        acc_y_filtered = self._moving_average(acc_y_corrected, window_size=filter_window)
        
        # 3. Detect candidate peaks.
        peaks, properties = find_peaks(acc_y_filtered, height=min_peak_height, prominence=min_peak_prominence)
        if len(peaks) < 2:
            print(f"Not enough candidate peaks detected (found {len(peaks)}).")
            self._plot_jump_detection(times, acc_y_filtered, peaks, properties)
            return []
        
        # 4. Pair candidate peaks to define jumps.
        jumps = []
        i = 0
        while i < len(peaks) - 1:
            takeoff_idx = peaks[i]
            landing_idx = peaks[i + 1]
            flight_time = times[landing_idx] - times[takeoff_idx]
            # Check if flight time is within plausible range.
            if min_flight_time <= flight_time <= max_flight_time:
                jump_height = (g * flight_time**2) / 8.0
                jumps.append({
                    'takeoff_time': times[takeoff_idx],
                    'landing_time': times[landing_idx],
                    'flight_time': flight_time,
                    'jump_height': jump_height
                })
                # Skip to next pair after landing.
                i += 2
            else:
                # If the pair does not yield a plausible flight time, move to next candidate.
                i += 1
        
        # 5. Plot for debugging.
        self._plot_peak_detection(times, acc_y_filtered, peaks, properties)
        
        return jumps

    def label_jumps(self, jumps: List[Tuple[float, float]]) -> List[str]:
        """
        Labels each detected jump with a landing type.
        :param jumps: List of jump time tuples.
        :return: List of labels corresponding to each jump.
        """
        # TODO: Implement the labeling logic based on additional sensor features.
        labels = ["Unknown" for _ in jumps]  # Placeholder implementation
        return labels

# Example usage:
if __name__ == '__main__':
    # Initialize the sensor object and load data from a JSON file.
    sensor = Sensor()
    file_path = "/Users/asmundur/Developer/MasterThesis/data/raw/7/Erik Öst/244730001982.json"
    sensor.load_json(file_path)
    
    if 0:
        # We set the start and end seconds for one run-up section with static time in the begining
        start_seconds = 212.9
        end_seconds = 219.1
        static_time = 2.0
        start_time = sensor.recording_data['timestamp'][0] + start_seconds
        end_time = sensor.recording_data['timestamp'][0] + end_seconds

        # Calculate the sprint speed and displacement
        sprint_speed, displacement = sensor.calculate_linear_sprint_speed(start_time, end_time, static_time)
        plotter = GraphPlotter(VelX=sprint_speed[:,0], VelY=sprint_speed[:,1], VelZ=sprint_speed[:,2])
        plotter.plot2d(title="Sprint Speed", xlabel="Time (s)", ylabel="Speed (m/s)")

        # Join the speed
        speed = np.linalg.norm(sprint_speed, axis=1)
        plotter = GraphPlotter(Speed=speed)
        plotter.plot2d(title="Speed", xlabel="Time (s)", ylabel="Speed (m/s)")
        print(f"Max speed: {np.max(speed):.2f} m/s")
        # Join the displacement
        total_displacement = np.linalg.norm(displacement, axis=1)
        plotter = GraphPlotter(Displacement=total_displacement)
        plotter.plot2d(title="Displacement", xlabel="Time (s)", ylabel="Displacement (m)")
        print(f"Total displacement: {total_displacement[-1]:.2f} m")

    if 0:
        # Calculate the jerk for both run-up and jump + landing
        start_seconds = 212.9
        end_seconds = 223.0
        start_time = sensor.recording_data['timestamp'][0] + start_seconds
        end_time = sensor.recording_data['timestamp'][0] + end_seconds
        jerk = sensor.calculate_jerk(start_time, end_time)
        plotter = GraphPlotter(Jerk=jerk)
        plotter.plot2d(title="Jerk", xlabel="Time (s)", ylabel="Jerk (m/s^3)")
        print(f"Max jerk: {np.max(jerk):.2f} m/s^3")

    if 0:
        # Calculate the impulse for both run-up and jump + landing
        start_seconds = 212.9
        end_seconds = 223.0
        static_time = 2.0
        start_time = sensor.recording_data['timestamp'][0] + start_seconds
        end_time = sensor.recording_data['timestamp'][0] + end_seconds
        filter_window = 5
        mass = 75.0  # kg
        impulse = sensor.calculate_impulse(start_time, end_time, mass, static_time, filter_window)
        plotter = GraphPlotter(Impulse=impulse)
        plotter.plot2d(title="Impulse", xlabel="Time (s)", ylabel="Impulse (Ns)")
        print(f"Max impulse: {np.max(impulse):.2f} Ns")

    if 0:
        # Calculate the kinetic energy at the end of trampette run-up
        # TODO: Adjust the start and end seconds based on the actual run-up section. Decide when we are interested in the kinetic energy.
        start_seconds = 212.9
        end_seconds = 219.1
        start_time = sensor.recording_data['timestamp'][0] + start_seconds
        end_time = sensor.recording_data['timestamp'][0] + end_seconds
        mass = 75.0  # kg
        kinetic_energy = sensor.calculate_kinetic_energy(start_time, end_time, mass, static_time)
        plotter = GraphPlotter(KineticEnergy=kinetic_energy)
        plotter.plot2d(title="Kinetic Energy", xlabel="Time (s)", ylabel="Energy (J)")
        print(f"Max kinetic energy: {np.max(kinetic_energy):.2f} J")

    if 1:
        # Calculate the jump height
        start_seconds = 215.9
        end_seconds = 224.0
        static_time = 2.0
        start_time = sensor.recording_data['timestamp'][0] + start_seconds
        end_time = sensor.recording_data['timestamp'][0] + end_seconds
        filter_window = 5
        min_peak_height=0.0,
        min_peak_prominence=1.0,
        jump_data = sensor.calculate_jump_height(start_time, end_time, static_time, filter_window, min_peak_height, min_peak_prominence)
        print(jump_data)

        # Calculate the potential energy at maximal jump height
        potential_energy = sensor.calculate_potential_energy_at_max_height(75.0, jump_data)
        print(f"Potential energy at max height: {potential_energy:.2f} J")

    if 0:
        # Calculate the rotational energy during the airborne phase
        start_seconds = 212.9
        end_seconds = 223.0
        static_time = 2.0
        start_time = sensor.recording_data['timestamp'][0] + start_seconds
        end_time = sensor.recording_data['timestamp'][0] + end_seconds
        body_position = "pike"
        static_time = 2.0
        filter_window = 5
        mass = 75.0
        height = 1.76
        times_air, rotational_energy = sensor.calculate_rotational_energy(start_time, end_time, body_position, static_time, filter_window, mass, height)
        plotter = GraphPlotter(RotationalEnergy=rotational_energy)
        plotter.plot2d(title="Rotational Energy", xlabel="Time (s)", ylabel="Energy (J)")
        print(f"Max rotational energy: {np.max(rotational_energy):.2f} J")

    if 0:
        # Calculate the number of jumps
        start_time = sensor.recording_data['timestamp'][0]
        end_time = sensor.recording_data['timestamp'][len(sensor.recording_data)-1]
        static_time = 2.0
        filter_window = 5
        min_peak_height = 40.0
        min_peak_prominence = 10.0
        g = 9.81
        min_flight_time = 0.3
        max_flight_time = 2.0
        jumps = sensor.detect_jumps(start_time, end_time, static_time, filter_window, min_peak_height, min_peak_prominence, g, min_flight_time, max_flight_time)
        print(f"Detected jumps: {len(jumps)}")