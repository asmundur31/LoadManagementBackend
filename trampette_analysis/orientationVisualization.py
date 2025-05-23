import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R
from ahrs.filters import Madgwick 
from utility import (
    read_json_file, 
)

# Example usage
file_path = "/Users/asmundur/Developer/MasterThesis/data/raw/5/Test with videos/203130000371.json"
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
start_seconds = 213#981.3
end_seconds = 217#984.8
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
gyro_array = np.deg2rad(gyro_array)
mag_array = np.column_stack((
    filtered_mag_data['magnX'],
    filtered_mag_data['magnY'],
    filtered_mag_data['magnZ']
))

dt = np.mean(np.diff(filtered_time_intervals))
print(f"Estimated sample period dt: {dt:.4f} s")

# Initialize the Madgwick filter
madgwick = Madgwick(dt=dt)
# Initialize quaternion (we assume the quaternion format is [q0, q1, q2, q3] with q0 as the scalar)
q = np.array([1.0, 0.0, 0.0, 0.0])

# Set up the matplotlib 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.ion()  # interactive mode on

def update_frame(i):
    global q
    # Get sensor measurements at time index i
    acc = acc_array[i]
    gyro = gyro_array[i]
    mag = mag_array[i]
    
    # Update orientation estimate using the Madgwick filter.
    # The filter returns a new quaternion estimate.
    q = madgwick.updateMARG(q, gyr=gyro, acc=acc, mag=mag)
    
    # Convert quaternion to rotation matrix.
    # Note: SciPy expects the quaternion in the format [x, y, z, w] where w is the scalar.
    # If your Madgwick filter outputs [q0, q1, q2, q3] with q0 as scalar, convert as follows:
    quat_scipy = np.array([q[1], q[2], q[3], q[0]])
    R_matrix = R.from_quat(quat_scipy).as_matrix()
    
    # Define the sensor frame unit vectors (in sensor coordinates)
    origin = np.array([0, 0, 0])
    sensor_x = np.array([1, 0, 0])
    sensor_y = np.array([0, 1, 0])
    sensor_z = np.array([0, 0, 1])
    
    # Rotate these vectors into the global frame
    x_global = R_matrix @ sensor_x
    y_global = R_matrix @ sensor_y
    z_global = R_matrix @ sensor_z
    
    # Clear the previous plot and redraw
    ax.clear()
    ax.quiver(*origin, *x_global, color="r", label="Sensor X")
    ax.quiver(*origin, *y_global, color="g", label="Sensor Y")
    ax.quiver(*origin, *z_global, color="b", label="Sensor Z")
    
    # Set plot limits and labels
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_xlabel("Global X")
    ax.set_ylabel("Global Y")
    ax.set_zlabel("Global Z")
    ax.set_title(f"Orientation at t = {filtered_time_intervals[i]:.2f}s")
    ax.legend()
    
    plt.draw()
    plt.pause(0.001)

# Create an animation that updates the sensor frame at each sample.
ani = FuncAnimation(fig, update_frame, frames=len(filtered_time_intervals), interval=dt*1000)
ani.save("orientation_animation.gif", writer="imagemagick", fps=30)
plt.show()