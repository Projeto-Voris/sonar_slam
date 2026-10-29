"""15-state Error-State Kalman Filter (ESKF) for 6-DoF ROV state estimation."""
from dataclasses import dataclass, field
import numpy as np
from scipy.stats import chi2

from ..geometry import (
    skew, exp_so3, rot_to_quat, quat_to_rot, quat_mult,
    quat_exp, rot_to_euler, wrap_angle
)


@dataclass
class EskfConfig:
    # Continuous-time noise spectral densities
    sigma_gyro: float = 0.002       # rad/s / sqrt(Hz)
    sigma_accel: float = 0.02       # m/s^2 / sqrt(Hz)
    sigma_gyro_bias: float = 1e-4   # rad/s^2 / sqrt(Hz)
    sigma_accel_bias: float = 1e-3  # m/s^3 / sqrt(Hz)

    # Measurement default standard deviations (when not in msg)
    sigma_dvl: float = 0.01         # m/s
    sigma_depth: float = 0.02       # m
    sigma_vis_pos: float = 0.05     # m
    sigma_vis_yaw: float = 0.02     # rad
    sigma_gravity: float = 0.5      # m/s^2 (deliberately weak)

    # Chi-square gating
    use_chi2_gate: bool = True
    gate_p_val: float = 0.99

    # Static gravity leveling thresholds
    static_gyro_threshold: float = 0.03      # rad/s
    static_accel_threshold: float = 0.2      # m/s^2 from g
    static_velocity_threshold: float = 0.05  # m/s
    enable_gravity_leveling: bool = True

    # Optional stationarity aiding requires fresh, accepted bottom-track DVL.
    enable_zupt: bool = False
    sigma_zupt: float = 0.02
    sigma_zero_gyro: float = 0.005
    zupt_min_duration: float = 0.5
    zupt_dvl_max_age: float = 0.3

    # Outlier / re-alignment thresholds
    dvl_max_velocity: float = 3.0            # m/s
    vis_realign_after_n_rejects: int = 15

    # Gravity constant
    gravity: float = 9.80665


class ESKF:
    """15-state Error-State Kalman Filter (nominal state 16, error state 15).
    Error state ordering:
      0:3   delta_p  (position, world frame)
      3:6   delta_v  (velocity, world frame)
      6:9   delta_th (rotation, body frame angle-axis)
      9:12  delta_bg (gyro bias, body frame)
      12:15 delta_ba (accel bias, body frame)
    """

    def __init__(self, config: EskfConfig = None):
        self.cfg = config or EskfConfig()

        # Nominal states
        self.p = np.zeros(3, dtype=np.float64)       # World frame position (Z-up)
        self.v = np.zeros(3, dtype=np.float64)       # World frame velocity
        self.q = np.array([1.0, 0.0, 0.0, 0.0])     # Body -> World quaternion
        self.R = np.eye(3, dtype=np.float64)        # Body -> World rotation matrix
        self.bg = np.zeros(3, dtype=np.float64)      # Gyro bias
        self.ba = np.zeros(3, dtype=np.float64)      # Accel bias

        # 15x15 Error covariance matrix
        self.P = np.diag([
            0.1**2, 0.1**2, 0.1**2,       # p
            0.05**2, 0.05**2, 0.05**2,    # v
            0.02**2, 0.02**2, 0.05**2,    # th (roll, pitch, yaw)
            (1e-3)**2, (1e-3)**2, (1e-3)**2,  # bg
            (1e-2)**2, (1e-2)**2, (1e-2)**2   # ba
        ]).astype(np.float64)

        self.g_world = np.array([0.0, 0.0, -self.cfg.gravity], dtype=np.float64)

        # Precompute chi2 gating thresholds (99%)
        self.chi2_gates = {
            1: chi2.ppf(self.cfg.gate_p_val, df=1),
            2: chi2.ppf(self.cfg.gate_p_val, df=2),
            3: chi2.ppf(self.cfg.gate_p_val, df=3),
            4: chi2.ppf(self.cfg.gate_p_val, df=4),
            6: chi2.ppf(self.cfg.gate_p_val, df=6),
        }

        # Visual odometry alignment state
        self.vis_initialized = False
        self.T_odom_vis = None
        self.vis_reject_count = 0

        # Last IMU timestamp
        self.last_imu_time = None
        self.dvl_age = float("inf")
        self.dvl_stationary = False
        self.stationary_duration = 0.0

    def initialize_attitude(self, accel_samples: np.ndarray):
        """Level initial roll/pitch from static accelerometer readings."""
        mean_accel = np.mean(accel_samples, axis=0)
        norm_a = np.linalg.norm(mean_accel)
        if norm_a < 1e-4:
            return
        # Accel measures upward reaction to gravity: a ~= -R^T * g_world
        # So in body frame: a_norm ~= [-sin(pitch), sin(roll)*cos(pitch), cos(roll)*cos(pitch)]
        an = mean_accel / norm_a
        pitch = np.arctan2(-an[0], np.sqrt(an[1]**2 + an[2]**2))
        roll = np.arctan2(an[1], an[2])
        yaw = 0.0

        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        self.R = Ry @ Rx
        self.q = rot_to_quat(self.R)

    def predict(self, gyro: np.ndarray, accel: np.ndarray, dt: float):
        """Strapdown mechanization and covariance propagation on raw IMU sample."""
        if dt <= 0.0 or dt > 0.5:
            return

        w = gyro - self.bg
        a_body = accel - self.ba

        # Acceleration in world frame
        a_world = self.R @ a_body + self.g_world

        # Nominal state propagation
        self.p += self.v * dt + 0.5 * a_world * (dt**2)
        self.v += a_world * dt

        # Attitude propagation (body-frame rotation vector)
        dq = quat_exp(w * dt)
        self.q = quat_mult(self.q, dq)
        self.q /= np.linalg.norm(self.q)
        self.R = quat_to_rot(self.q)

        # Discrete error state transition matrix Phi = I + F*dt + 0.5*(F*dt)^2
        F = np.zeros((15, 15), dtype=np.float64)
        F[0:3, 3:6] = np.eye(3)
        F[3:6, 6:9] = -self.R @ skew(a_body)
        F[3:6, 12:15] = -self.R
        F[6:9, 6:9] = -skew(w)
        F[6:9, 9:12] = -np.eye(3)

        F_dt = F * dt
        Phi = np.eye(15) + F_dt + 0.5 * (F_dt @ F_dt)

        # Process noise covariance Qd
        G = np.zeros((15, 12), dtype=np.float64)
        G[3:6, 0:3] = -self.R          # accel noise
        G[6:9, 3:6] = -np.eye(3)        # gyro noise
        G[9:12, 6:9] = np.eye(3)        # gyro bias walk
        G[12:15, 9:12] = np.eye(3)      # accel bias walk

        Q_cont = np.diag([
            self.cfg.sigma_accel**2, self.cfg.sigma_accel**2, self.cfg.sigma_accel**2,
            self.cfg.sigma_gyro**2, self.cfg.sigma_gyro**2, self.cfg.sigma_gyro**2,
            self.cfg.sigma_gyro_bias**2, self.cfg.sigma_gyro_bias**2, self.cfg.sigma_gyro_bias**2,
            self.cfg.sigma_accel_bias**2, self.cfg.sigma_accel_bias**2, self.cfg.sigma_accel_bias**2
        ])
        Qd = G @ Q_cont @ G.T * dt

        # Propagate covariance
        self.P = Phi @ self.P @ Phi.T + Qd
        self.P = 0.5 * (self.P + self.P.T)  # Keep symmetric

        self.dvl_age += dt
        stationary = (self.dvl_stationary and
                      self.dvl_age <= self.cfg.zupt_dvl_max_age and
                      np.linalg.norm(gyro - self.bg) < self.cfg.static_gyro_threshold and
                      abs(np.linalg.norm(accel - self.ba) - self.cfg.gravity) < self.cfg.static_accel_threshold)
        self.stationary_duration = self.stationary_duration + dt if stationary else 0.0
        if self.cfg.enable_zupt and stationary and self.stationary_duration >= self.cfg.zupt_min_duration:
            self.correct_zero_motion(gyro)

        # Quasi-static gravity leveling when stationary
        if self.cfg.enable_gravity_leveling:
            self._maybe_gravity_leveling(gyro, accel)

    def correct_zero_motion(self, gyro: np.ndarray) -> bool:
        """Fuse zero world velocity and zero angular rate; never hard-reset states."""
        H = np.zeros((6, 15))
        H[:3, 3:6] = np.eye(3)
        H[3:, 9:12] = np.eye(3)
        residual = np.concatenate((-self.v, gyro - self.bg))
        noise = np.diag([self.cfg.sigma_zupt**2] * 3 + [self.cfg.sigma_zero_gyro**2] * 3)
        return self._apply_correction(residual, H, noise, dof=6)

    def _maybe_gravity_leveling(self, gyro: np.ndarray, accel: np.ndarray):
        """Static gravity aiding bounds roll/pitch drift when vehicle is still."""
        w_norm = np.linalg.norm(gyro)
        v_norm = np.linalg.norm(self.v)
        a_diff = np.abs(np.linalg.norm(accel) - self.cfg.gravity)

        if (w_norm < self.cfg.static_gyro_threshold and
            v_norm < self.cfg.static_velocity_threshold and
            a_diff < self.cfg.static_accel_threshold):

            # Accelerometer measures -R^T * g_world
            expected_a = self.R.T @ (-self.g_world)
            residual = accel - self.ba - expected_a

            H = np.zeros((3, 15), dtype=np.float64)
            # Jacobian w.r.t delta_th (body frame): d/d(delta_th) (exp(-delta_th) R^T (-g)) = [R^T (-g)]x
            H[:, 6:9] = skew(expected_a)

            R_cov = np.eye(3) * (self.cfg.sigma_gravity**2)
            self._apply_correction(residual, H, R_cov, dof=3)

    def correct_dvl(self, v_dvl_meas: np.ndarray, R_bd: np.ndarray, r_bd: np.ndarray,
                    cov: np.ndarray = None, gyro: np.ndarray = None) -> bool:
        """DVL body velocity correction with lever arm compensation.
        v_dvl = R_bd^T (R^T v + w x r_bd)
        """
        self.dvl_stationary = False
        if not np.all(np.isfinite(v_dvl_meas)):
            return False
        if np.linalg.norm(v_dvl_meas) > self.cfg.dvl_max_velocity:
            return False

        w = (gyro - self.bg) if gyro is not None else np.zeros(3)
        v_body = self.R.T @ self.v
        v_dvl_pred = R_bd.T @ (v_body + np.cross(w, r_bd))

        residual = v_dvl_meas - v_dvl_pred

        # Jacobians:
        # H_v = R_bd^T * R^T
        # H_th = R_bd^T * [R^T v]x
        # H_bg = R_bd^T * [r_bd]x
        H = np.zeros((3, 15), dtype=np.float64)
        H[:, 3:6] = R_bd.T @ self.R.T
        H[:, 6:9] = R_bd.T @ skew(v_body)
        H[:, 9:12] = R_bd.T @ skew(r_bd)

        if cov is None or np.all(cov == 0):
            R_cov = np.eye(3) * (self.cfg.sigma_dvl**2)
        else:
            R_cov = cov.copy()

        accepted = self._apply_correction(residual, H, R_cov, dof=3)
        if accepted:
            self.dvl_age = 0.0
            self.dvl_stationary = np.linalg.norm(R_bd @ v_dvl_meas - np.cross(w, r_bd)) < self.cfg.static_velocity_threshold
        return accepted

    def correct_depth(self, depth_z_up: float, r_depth: np.ndarray, sigma: float = None) -> bool:
        """Depth / altitude correction with lever arm compensation.
        depth_z_up: vehicle depth in world frame (Z-up positive)
        """
        # Sensor world position Z: p_z + (R * r_depth)_z
        pred_z = self.p[2] + (self.R @ r_depth)[2]
        residual = np.array([depth_z_up - pred_z])

        H = np.zeros((1, 15), dtype=np.float64)
        H[0, 2] = 1.0  # d(p_z)/d(delta_p_z)
        # d(R * exp(delta_th) * r_depth)_z / d(delta_th) = -(R * [r_depth]x)_z
        H[0, 6:9] = -(self.R @ skew(r_depth))[2, :]

        s = sigma if (sigma is not None and sigma > 0) else self.cfg.sigma_depth
        R_cov = np.array([[s**2]])

        return self._apply_correction(residual, H, R_cov, dof=1)

    def correct_visual_pose(self, p_vis: np.ndarray, yaw_vis: float,
                            p_cov: np.ndarray = None, yaw_cov: float = None) -> bool:
        """Visual SLAM 6-DoF pose aiding (x, y, z, yaw) with one-time frame anchoring."""
        if not self.vis_initialized:
            # Anchor visual frame onto current odom frame
            # T_odom_vis translation and yaw offset
            current_yaw = rot_to_euler(self.R)[2]
            self.vis_yaw_offset = wrap_angle(current_yaw - yaw_vis)
            R_align = exp_so3(np.array([0, 0, self.vis_yaw_offset]))
            self.vis_pos_offset = self.p - R_align @ p_vis
            self.vis_initialized = True
            return True

        # Transform incoming visual pose into odom frame via stored alignment
        R_align = exp_so3(np.array([0, 0, self.vis_yaw_offset]))
        p_aligned = self.vis_pos_offset + R_align @ p_vis
        yaw_aligned = wrap_angle(yaw_vis + self.vis_yaw_offset)

        # Residuals
        current_yaw = rot_to_euler(self.R)[2]
        res_p = p_aligned - self.p
        res_yaw = wrap_angle(yaw_aligned - current_yaw)
        residual = np.array([res_p[0], res_p[1], res_p[2], res_yaw])

        H = np.zeros((4, 15), dtype=np.float64)
        H[0:3, 0:3] = np.eye(3)
        # Yaw derivative w.r.t body angle error delta_th:
        # In ENU: yaw = atan2(R[1,0], R[0,0])
        roll, pitch, _ = rot_to_euler(self.R)
        cp = np.cos(pitch)
        if np.abs(cp) > 1e-4:
            H[3, 6:9] = np.array([0.0, np.sin(roll) / cp, np.cos(roll) / cp])
        else:
            H[3, 8] = 1.0

        # Measurement noise
        sig_p = self.cfg.sigma_vis_pos
        sig_yaw = self.cfg.sigma_vis_yaw
        R_cov = np.diag([sig_p**2, sig_p**2, sig_p**2, sig_yaw**2])
        if p_cov is not None and p_cov.shape == (3, 3):
            R_cov[0:3, 0:3] = p_cov
        if yaw_cov is not None and yaw_cov > 0:
            R_cov[3, 3] = yaw_cov

        success = self._apply_correction(residual, H, R_cov, dof=4)
        if not success:
            self.vis_reject_count += 1
            if self.vis_reject_count >= self.cfg.vis_realign_after_n_rejects:
                # Re-align
                self.vis_initialized = False
                self.vis_reject_count = 0
        else:
            self.vis_reject_count = 0
        return success

    def _apply_correction(self, residual: np.ndarray, H: np.ndarray,
                          R_cov: np.ndarray, dof: int) -> bool:
        """Numerically robust Joseph-form Kalman update with Chi-squared gating."""
        if not (np.all(np.isfinite(residual)) and np.all(np.isfinite(R_cov))):
            return False
        if np.min(np.linalg.eigvalsh(R_cov)) <= 0:
            return False
        S = H @ self.P @ H.T + R_cov
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False

        # Chi-squared Mahalanobis distance test
        if self.cfg.use_chi2_gate and dof in self.chi2_gates:
            maha_d2 = float(residual.T @ S_inv @ residual)
            if maha_d2 > self.chi2_gates[dof]:
                return False

        # Kalman gain
        K = self.P @ H.T @ S_inv

        # Error state correction
        dx = K @ residual

        # Update nominal states
        self.p += dx[0:3]
        self.v += dx[3:6]

        dth = dx[6:9]
        dq = quat_exp(dth)
        self.q = quat_mult(self.q, dq)
        self.q /= np.linalg.norm(self.q)
        self.R = quat_to_rot(self.q)

        self.bg += dx[9:12]
        self.ba += dx[12:15]

        # Joseph form covariance update: P = (I - K*H)*P*(I - K*H)^T + K*R*K^T
        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_cov @ K.T
        self.P = 0.5 * (self.P + self.P.T)  # Ensure symmetry
        return True

    def get_pose_covariance_6x6(self) -> np.ndarray:
        """Extract 6x6 pose covariance [x, y, z, roll, pitch, yaw] in world frame."""
        cov6 = np.zeros((6, 6), dtype=np.float64)
        cov6[0:3, 0:3] = self.P[0:3, 0:3]

        # Body rotation covariance transformed to world
        cov_th_body = self.P[6:9, 6:9]
        cov_th_world = self.R @ cov_th_body @ self.R.T
        cov6[3:6, 3:6] = cov_th_world
        cov6[0:3, 3:6] = self.P[0:3, 6:9] @ self.R.T
        cov6[3:6, 0:3] = cov6[0:3, 3:6].T
        return cov6
