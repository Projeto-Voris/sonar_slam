"""Geometry and rotation helpers (SO(3) and SE(3)) in pure NumPy."""
import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    """Return 3x3 skew-symmetric matrix [v]x."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=np.float64)


def wrap_angle(theta: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def exp_so3(w: np.ndarray) -> np.ndarray:
    """Rodrigues formula for exponential map so(3) -> SO(3)."""
    theta = np.linalg.norm(w)
    if theta < 1e-10:
        return np.eye(3) + skew(w)
    axis = w / theta
    K = skew(axis)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def log_so3(R: np.ndarray) -> np.ndarray:
    """Logarithmic map SO(3) -> so(3)."""
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-10:
        return np.zeros(3)
    sin_theta = np.sin(theta)
    if sin_theta < 1e-10:
        # Near pi
        diag = np.maximum(0.0, (np.diag(R) + 1.0) / 2.0)
        axis = np.sqrt(diag)
        if R[0, 1] < 0:
            axis[1] = -axis[1]
        if R[0, 2] < 0:
            axis[2] = -axis[2]
        return theta * axis
    w = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ]) / (2.0 * sin_theta)
    return theta * w


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [qw, qx, qy, qz]."""
    tr = np.trace(R)
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    return q / np.linalg.norm(q)


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [qw, qx, qy, qz] to 3x3 rotation matrix."""
    qw, qx, qy, qz = q / np.linalg.norm(q)
    return np.array([
        [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
        [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
        [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)]
    ], dtype=np.float64)


def quat_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions [qw, qx, qy, qz]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ], dtype=np.float64)


def quat_exp(w: np.ndarray) -> np.ndarray:
    """Convert small rotation vector w to quaternion [qw, qx, qy, qz]."""
    theta = np.linalg.norm(w)
    if theta < 1e-10:
        return np.array([1.0, 0.5 * w[0], 0.5 * w[1], 0.5 * w[2]], dtype=np.float64)
    axis = w / theta
    half_theta = 0.5 * theta
    return np.array([np.cos(half_theta), *(np.sin(half_theta) * axis)], dtype=np.float64)


def rot_to_euler(R: np.ndarray) -> np.ndarray:
    """Extract roll, pitch, yaw (XYZ intrinsic / ZYX extrinsic) from rotation matrix."""
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    if np.abs(np.abs(pitch) - np.pi / 2.0) < 1e-6:
        # Gimbal lock
        roll = 0.0
        yaw = np.arctan2(R[0, 1], R[1, 1])
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([wrap_angle(roll), wrap_angle(pitch), wrap_angle(yaw)])


def euler_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert roll, pitch, yaw to 3x3 rotation matrix."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])

    return Rz @ Ry @ Rx


def make_se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Construct 4x4 SE(3) transformation matrix."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inv_se3(T: np.ndarray) -> np.ndarray:
    """Invert 4x4 SE(3) matrix."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv
