import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def takeoff_and_landing_detection(jump_imu_df: pd.DataFrame) -> tuple:
    # Extract the y-axis acceleration data (assuming lower_back_accy is vertical acceleration)
    acc_lb_y = jump_imu_df['lower_back_accy'].to_numpy()

    # Time difference (sampling rate assumed to be 104 Hz, or dt = 1/104 seconds)
    dt = 1/104.0

    # Initialize vertical velocity array
    vert_velocity = np.zeros_like(acc_lb_y)

    # Integrate acceleration to get velocity
    for i in range(1, len(acc_lb_y)):
        vert_velocity[i] = vert_velocity[i-1] + acc_lb_y[i] * dt

    # Detect takeoff and landing phases
    takeoff_mask = (acc_lb_y[1:] > 0.5) & (acc_lb_y[:-1] <= 0.5)
    landing_mask = (acc_lb_y[1:] < -0.5) & (acc_lb_y[:-1] >= -0.5)

    # Get the indices where takeoff and landing occur
    takeoff_indices = np.where(takeoff_mask)[0] + 1  # +1 to align with the second element
    landing_indices = np.where(landing_mask)[0] + 1  # +1 to align with the second element

    # Extract timestamps for takeoff and landing events
    takeoff_timestamps = jump_imu_df['timestamp'].iloc[takeoff_indices].values
    landing_timestamps = jump_imu_df['timestamp'].iloc[landing_indices].values

    # Plot the vertical velocity
    plt.figure(figsize=(10, 6))
    plt.plot(jump_imu_df['timestamp'], vert_velocity, label='Vertical Velocity', color='b')
    plt.axhline(y=0, color='k', linestyle='--')  # Zero velocity line

    # Highlight takeoff and landing points
    plt.scatter(takeoff_timestamps, vert_velocity[takeoff_indices], color='g', label='Takeoff', zorder=5)
    plt.scatter(landing_timestamps, vert_velocity[landing_indices], color='r', label='Landing', zorder=5)

    # Labeling and formatting
    plt.title('Vertical Velocity vs Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Vertical Velocity (m/s)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Show plot
    plt.show()

    # Return timestamps of takeoff and landing
    return takeoff_timestamps, landing_timestamps