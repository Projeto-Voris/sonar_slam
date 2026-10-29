"""Complementary filter / Mahony AHRS + DVL dead-reckoning baseline."""
from dataclasses import dataclass
import numpy as np

from ..geometry import skew, exp_so3, rot_to_quat, quat_to_rot, quat_mult, quat_exp


@dataclass
class ComplementaryConfig:
    kp: float = 2.0                 # Mahony proportional gain
    ki: float = 0.05                # Mahony integral gain (gyro bias estimation)
    accel_gate: float = 1.0         # Max accel norm deviation from gravity (m/s^2)
    gravity: float = 9.80665


class ComplementaryFilter:
    """Lightweight Mahony AHRS attitude observer with DVL trapezoidal integration."""

    def __init__(self, config: ComplementaryConfig = None):
        self.cfg = config or ComplementaryConfig()

        self.p = np.zeros(3, dtype=np.float64)       # Position in world (FLU/ENU Z-up)
        self.v = np.zeros(3, dtype=np.float64)       # Velocity in world
        self.q = np.array([1.0, 0.0, 0.0, 0.0])     # Body -> World attitude
        self.R = np.eye(3, dtype=np.float64)
        self.bg = np.zeros(3, dtype=np.float64)      # Estimated gyro bias
        self.prev_v_world = np.zeros(3, dtype=np.float64)

    def initialize_attitude(self, accel: np.ndarray):
        """Initial leveling from static gravity reading."""
        accel = np.asarray(accel, dtype=float)
        if accel.ndim == 2:
            accel = np.mean(accel, axis=0)
        norm_a = np.linalg.norm(accel)
        if norm_a < 1e-4:
            return
        an = accel / norm_a
        pitch = np.arctan2(-an[0], np.sqrt(an[1]**2 + an[2]**2))
        roll = np.arctan2(an[1], an[2])

        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        self.R = Ry @ Rx
        self.q = rot_to_quat(self.R)

    def update_imu(self, gyro: np.ndarray, accel: np.ndarray, dt: float):
        """Mahony AHRS step."""
        if dt <= 0.0 or dt > 0.5:
            return

        w_meas = gyro.copy()
        norm_a = np.linalg.norm(accel)

        # Accelerometer correction vector
        e_rot = np.zeros(3, dtype=np.float64)
        if np.abs(norm_a - self.cfg.gravity) < self.cfg.accel_gate and norm_a > 1e-4:
            a_norm = accel / norm_a
            # Expected gravity direction in body frame: R^T * [0, 0, 1]
            g_b = self.R.T @ np.array([0.0, 0.0, 1.0])
            # Cross product of measured and expected gravity gives rotation error
            e_rot = np.cross(a_norm, g_b)

        # Integral gain updates gyro bias estimate
        self.bg -= self.cfg.ki * e_rot * dt

        # Corrected angular velocity
        w = (w_meas - self.bg) + self.cfg.kp * e_rot

        # Attitude integration
        dq = quat_exp(w * dt)
        self.q = quat_mult(self.q, dq)
        self.q /= np.linalg.norm(self.q)
        self.R = quat_to_rot(self.q)

    def update_dvl(self, v_dvl_meas: np.ndarray, R_bd: np.ndarray, dt: float):
        """Integrate DVL velocity."""
        if dt <= 0.0 or dt > 1.0:
            return
        v_body = R_bd @ v_dvl_meas
        v_world = self.R @ v_body
        self.v = v_world
        # Trapezoidal integration for position
        self.p += 0.5 * (self.prev_v_world + v_world) * dt
        self.prev_v_world = v_world

    def update_depth(self, depth_z_up: float, r_depth: np.ndarray):
        """Direct depth correction with lever arm."""
        sensor_z_offset = (self.R @ r_depth)[2]
        self.p[2] = depth_z_up - sensor_z_offset
