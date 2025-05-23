import numpy as np
from scipy.spatial.transform import Rotation as R

def normalize_quaternion(q):
    return q / np.linalg.norm(q)

def skew(w):
    return np.array([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0]
    ])

def quaternion_to_rotation_matrix(q):
    return R.from_quat(q).as_matrix()

def quaternion_multiply(q1, q2):
    return R.from_quat(q1) * R.from_quat(q2)

def small_angle_quat(delta_theta):
    theta = np.linalg.norm(delta_theta)
    dq = np.hstack([
        np.cos(theta / 2),
        np.sin(theta / 2) * delta_theta / theta if theta > 1e-8 else [0, 0, 0]
    ])
    return dq

def gravity_in_sensor_frame(q):
    Rg = quaternion_to_rotation_matrix(q)
    return Rg.T @ np.array([0, 0, -9.81])  # Global gravity

# Initial state
q = np.array([1.0, 0.0, 0.0, 0.0])  # initial orientation quaternion (w, x, y, z)
bias = np.zeros(3)  # gyro bias
P = np.eye(6) * 0.01  # Covariance for [theta_error, bias]

# Tuning parameters
gyro_noise = 0.01
acc_noise_static = 0.1
acc_noise_dynamic = 10.0
bias_noise = 1e-5
dt = 0.01  # time step (adjust for your data)

def ekf_orientation_step(acc, gyro, is_static, q, bias, P):
    # --- Predict ---
    omega = gyro - bias
    omega_norm = np.linalg.norm(omega)
    delta_q = R.from_rotvec(omega * dt).as_quat()
    q = R.from_quat(q) * R.from_quat(delta_q)
    q = q.as_quat()

    # Linearize
    F = np.eye(6)
    F[0:3, 3:6] = -np.eye(3) * dt

    Q = np.diag([gyro_noise**2]*3 + [bias_noise**2]*3)
    P = F @ P @ F.T + Q

    # --- Update using accelerometer ---
    g_body = gravity_in_sensor_frame(q)
    acc = acc / np.linalg.norm(acc) * 9.81
    y = acc - g_body

    # Jacobian H: partial derivative of g_body wrt theta_error (approx.)
    H = np.zeros((3, 6))
    H[0:3, 0:3] = -skew(g_body)

    Rk = np.eye(3) * (acc_noise_static if is_static else acc_noise_dynamic)

    S = H @ P @ H.T + Rk
    K = P @ H.T @ np.linalg.inv(S)
    delta_x = K @ y
    theta_corr = delta_x[0:3]
    bias_corr = delta_x[3:6]

    # Correct orientation
    dq = R.from_rotvec(theta_corr).as_quat()
    q = R.from_quat(dq) * R.from_quat(q)
    q = normalize_quaternion(q.as_quat())

    # Update bias and covariance
    bias += bias_corr
    P = (np.eye(6) - K @ H) @ P

    return q, bias, P
