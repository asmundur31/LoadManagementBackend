import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R

def visualize_puck_and_imu_df(
    quaternions: np.ndarray,
    imu_df: pd.DataFrame,
    columns: list,
    timestamp_col: str = None,
    interval: int = 5,
):
    if quaternions.shape[1] != 4:
        raise ValueError("Quaternions must be Nx4.")
    
    imu_data = imu_df[columns].values
    timestamps = imu_df[timestamp_col].values if timestamp_col else np.arange(len(imu_df)) / 100

    fig = plt.figure(figsize=(12, 6))
    ax3d = fig.add_subplot(121, projection='3d')
    ax2d = fig.add_subplot(122)

    # Puck geometry
    radius, height = 1.0, 0.2
    theta = np.linspace(0, 2 * np.pi, 30)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z_top = np.full_like(x, height / 2)
    z_bottom = -z_top
    top = np.array([x, y, z_top]).T
    bottom = np.array([x, y, z_bottom]).T[::-1]
    sides = [[top[i], top[(i+1)%len(top)], bottom[(i+1)%len(bottom)], bottom[i]] for i in range(len(top))]

    ax3d.set_xlim([-1.5, 1.5])
    ax3d.set_ylim([-1.5, 1.5])
    ax3d.set_zlim([-1.5, 1.5])
    ax3d.set_title("3D Orientation")
    ax3d.set_box_aspect([1, 1, 1])
    ax3d.view_init(elev=20, azim=30)

    top_poly = Poly3DCollection([top], facecolor='lightblue', alpha=0.7)
    bottom_poly = Poly3DCollection([bottom], facecolor='steelblue', alpha=0.7)
    side_polys = Poly3DCollection(sides, facecolor='skyblue', alpha=0.4)
    ax3d.add_collection3d(top_poly)
    ax3d.add_collection3d(bottom_poly)
    ax3d.add_collection3d(side_polys)

    origin = np.array([0, 0, 0])
    axis_length = 1.0
    arrows = [
        ax3d.quiver(*origin, 1, 0, 0, color='r', linewidth=2),
        ax3d.quiver(*origin, 0, 1, 0, color='g', linewidth=2),
        ax3d.quiver(*origin, 0, 0, 1, color='b', linewidth=2),
    ]

    # IMU data plot
    colors = ['r', 'g', 'b']  # Red for x, green for y, blue for z
    lines = [ax2d.plot([], [], label=label, color=colors[i])[0] for i, label in enumerate(columns)]


    ax2d.set_xlim([timestamps[0], timestamps[min(len(timestamps)-1, 200)]])
    ax2d.set_ylim([
        np.min(imu_data[:, :len(columns)]) * 1.2,
        np.max(imu_data[:, :len(columns)]) * 1.2
    ])
    ax2d.set_title("Raw IMU Data")
    ax2d.set_xlabel("Time (s)" if timestamp_col else "Index")
    ax2d.set_ylabel("Value")
    ax2d.legend()
    ax2d.grid(True)

    frame_index = [0]
    playing = [True]

    def update(frame):
        rot = R.from_quat(quaternions[frame], scalar_first=True)
        rot_matrix = rot.as_matrix()

        def transform(vertices):
            return (rot_matrix @ vertices.T).T

        top_rot = transform(top)
        bottom_rot = transform(bottom)
        sides_rot = [transform(np.array(face)) for face in sides]
        top_poly.set_verts([top_rot])
        bottom_poly.set_verts([bottom_rot])
        side_polys.set_verts(sides_rot)

        window_size = 200
        start = max(0, frame - window_size)
        for i, line in enumerate(lines):
            line.set_data(timestamps[start:frame], imu_data[start:frame, i])
        ax2d.set_xlim([timestamps[start], timestamps[frame]])

        # Rotate the coordinate axes
        x_rot = rot_matrix @ np.array([axis_length, 0, 0])
        y_rot = rot_matrix @ np.array([0, axis_length, 0])
        z_rot = rot_matrix @ np.array([0, 0, axis_length])

        for axis in arrows:
            axis.remove()

        # Re-create the arrows and store them in a list
        arrows[:] = [
            ax3d.quiver(0, 0, 0, *x_rot, color='g', linewidth=2),
            ax3d.quiver(0, 0, 0, *y_rot, color='b', linewidth=2),
            ax3d.quiver(0, 0, 0, *z_rot, color='r', linewidth=2),
        ]

    def on_key(event):
        if event.key == 'right':
            frame_index[0] = min(frame_index[0] + 100, len(quaternions) - 1)
            update(frame_index[0])
            fig.canvas.draw_idle()
        elif event.key == 'left':
            frame_index[0] = max(frame_index[0] - 100, 0)
            update(frame_index[0])
            fig.canvas.draw_idle()
        elif event.key == ' ':
            playing[0] = not playing[0]

    def timer_callback(*args):
        if playing[0]:
            frame_index[0] = (frame_index[0] + 1) % len(quaternions)
            update(frame_index[0])
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key)
    timer = fig.canvas.new_timer(interval=interval)
    timer.add_callback(timer_callback)
    timer.start()

    update(0)
    plt.tight_layout()
    plt.show()
