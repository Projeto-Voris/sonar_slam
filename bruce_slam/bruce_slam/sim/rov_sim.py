"""6-DoF ROV trajectory and underwater sensor simulator."""
from dataclasses import dataclass, field
from typing import Tuple
import numpy as np

from ..geometry import euler_to_rot, rot_to_quat, make_se3, inv_se3


@dataclass
class SimConfig:
    duration: float = 60.0          # Simulation duration (s)
    # Sensor frequencies
    imu_rate: float = 100.0         # Hz
    depth_rate: float = 20.0        # Hz
    dvl_rate: float = 10.0          # Hz
    stereo_rate: float = 10.0       # Hz
    sonar_rate: float = 8.0         # Hz

    # Trajectory parameters (elliptical loop)
    radius_x: float = 5.0
    radius_y: float = 3.0
    speed: float = 0.5              # m/s
    depth_nominal: float = 2.0      # m (positive down)

    # Noise parameters
    gyro_noise: float = 0.002       # rad/s
    accel_noise: float = 0.02       # m/s^2
    dvl_noise: float = 0.01         # m/s
    depth_noise: float = 0.01       # m
    stereo_noise_pos: float = 0.02  # m
    stereo_drift_rate: float = 0.001# rad/s heading drift
    sonar_noise: float = 0.02       # m
    sonar_outlier_ratio: float = 0.01

    # Lever arms (relative to base_link)
    r_imu: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    r_dvl: np.ndarray = field(default_factory=lambda: np.array([0.1, 0.0, -0.2]))
    r_depth: np.ndarray = field(default_factory=lambda: np.array([-0.2, 0.0, 0.1]))
    r_stereo: np.ndarray = field(default_factory=lambda: np.array([0.3, 0.1, 0.0]))
    r_sonar: np.ndarray = field(default_factory=lambda: np.array([0.4, 0.0, -0.1]))


class ROVSimulator:
    """Generates ground truth vehicle trajectory and noisy sensor readings."""

    def __init__(self, config: SimConfig = None):
        self.cfg = config or SimConfig()
        self._build_scene()

        # Biases
        self.bg_true = np.array([0.001, -0.001, 0.002])
        self.ba_true = np.array([0.01, -0.02, 0.01])
        self.stereo_drift = 0.0

    def _build_scene(self):
        """Construct synthetic underwater 3-D structure (floor, walls, pillar, pipe)."""
        pts = []
        # 1. Floor at z = -3.5 (in ENU world frame)
        fx = np.linspace(-8.0, 8.0, 60)
        fy = np.linspace(-6.0, 6.0, 50)
        FX, FY = np.meshgrid(fx, fy)
        FZ = np.full_like(FX, -3.5)
        pts.append(np.vstack((FX.ravel(), FY.ravel(), FZ.ravel())).T)

        # 2. Wall at x = 7.0
        wy = np.linspace(-6.0, 6.0, 40)
        wz = np.linspace(-3.5, 0.0, 20)
        WY, WZ = np.meshgrid(wy, wz)
        WX = np.full_like(WY, 7.0)
        pts.append(np.vstack((WX.ravel(), WY.ravel(), WZ.ravel())).T)

        # 3. Cylindrical pillar at (2.0, 2.0)
        theta = np.linspace(0, 2 * np.pi, 30)
        pz = np.linspace(-3.5, 0.0, 20)
        TH, PZ = np.meshgrid(theta, pz)
        radius = 0.8
        PX = 2.0 + radius * np.cos(TH)
        PY = 2.0 + radius * np.sin(TH)
        pts.append(np.vstack((PX.ravel(), PY.ravel(), PZ.ravel())).T)

        self.scene_points = np.vstack(pts)

    def get_ground_truth(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (p, v, a, R, w) at time t."""
        # 1-second still phase for static leveling initialization
        if t < 1.0:
            p = np.array([0.0, 0.0, -self.cfg.depth_nominal])
            v = np.zeros(3)
            a = np.zeros(3)
            R = np.eye(3)
            w = np.zeros(3)
            return p, v, a, R, w

        t_eff = t - 1.0
        omega = self.cfg.speed / np.sqrt(self.cfg.radius_x * self.cfg.radius_y)

        # Ellipse: x = A*sin(wt), y = B*(1 - cos(wt))
        x = self.cfg.radius_x * np.sin(omega * t_eff)
        y = self.cfg.radius_y * (1.0 - np.cos(omega * t_eff))
        z = -self.cfg.depth_nominal + 0.2 * np.sin(0.5 * omega * t_eff)
        p = np.array([x, y, z])

        vx = self.cfg.radius_x * omega * np.cos(omega * t_eff)
        vy = self.cfg.radius_y * omega * np.sin(omega * t_eff)
        vz = 0.2 * 0.5 * omega * np.cos(0.5 * omega * t_eff)
        v = np.array([vx, vy, vz])

        ax = -self.cfg.radius_x * (omega**2) * np.sin(omega * t_eff)
        ay = self.cfg.radius_y * (omega**2) * np.cos(omega * t_eff)
        az = -0.2 * (0.5 * omega)**2 * np.sin(0.5 * omega * t_eff)
        a = np.array([ax, ay, az])

        # Heading aligns with velocity vector
        yaw = np.arctan2(vy, vx)
        pitch = 0.05 * np.sin(omega * t_eff)
        roll = 0.02 * np.cos(omega * t_eff)
        R = euler_to_rot(roll, pitch, yaw)

        # Angular velocity
        w = np.array([
            -0.02 * omega * np.sin(omega * t_eff),
            0.05 * omega * np.cos(omega * t_eff),
            omega
        ])
        return p, v, a, R, w

    def sample_imu(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate gyro (rad/s) and accel (m/s^2) in body frame."""
        _, _, a_world, R, w_body = self.get_ground_truth(t)
        g_world = np.array([0.0, 0.0, -9.80665])

        # Specific force: f = R^T * (a_world - g_world)
        f_body = R.T @ (a_world - g_world)
        accel = f_body + self.ba_true + np.random.normal(0, self.cfg.accel_noise, 3)
        gyro = w_body + self.bg_true + np.random.normal(0, self.cfg.gyro_noise, 3)
        return gyro, accel

    def sample_dvl(self, t: float) -> np.ndarray:
        """Generate DVL velocity in dvl_link frame."""
        _, v_world, _, R, w_body = self.get_ground_truth(t)
        v_body = R.T @ v_world
        # Lever arm effect: v_sensor = v_body + w x r
        v_dvl_pt = v_body + np.cross(w_body, self.cfg.r_dvl)
        v_meas = v_dvl_pt + np.random.normal(0, self.cfg.dvl_noise, 3)
        return v_meas

    def sample_depth(self, t: float) -> float:
        """Generate positive-down depth in meters."""
        p_world, _, _, R, _ = self.get_ground_truth(t)
        p_sensor_z = p_world[2] + (R @ self.cfg.r_depth)[2]
        # Depth is positive down: depth = -z
        depth_meas = -p_sensor_z + np.random.normal(0, self.cfg.depth_noise)
        return float(depth_meas)

    def sample_stereo(self, t: float, dt: float) -> Tuple[np.ndarray, float]:
        """Generate stereo (p_base, yaw_base) with drift."""
        p_world, _, _, R, _ = self.get_ground_truth(t)
        self.stereo_drift += self.cfg.stereo_drift_rate * dt

        # Visual yaw and position
        from ..geometry import rot_to_euler
        _, _, yaw_true = rot_to_euler(R)
        yaw_meas = yaw_true + self.stereo_drift + np.random.normal(0, 0.01)
        p_meas = p_world + np.random.normal(0, self.cfg.stereo_noise_pos, 3)
        return p_meas, float(yaw_meas)

    def sample_sonar(self, t: float) -> np.ndarray:
        """Simulate 3-D forward-looking sonar scan in sonar_link frame."""
        p_world, _, _, R, _ = self.get_ground_truth(t)
        T_world_base = make_se3(R, p_world)
        T_base_sonar = make_se3(np.eye(3), self.cfg.r_sonar)
        T_world_sonar = T_world_base @ T_base_sonar
        T_sonar_world = inv_se3(T_world_sonar)

        # Transform scene into sonar frame
        R_sw = T_sonar_world[:3, :3]
        t_sw = T_sonar_world[:3, 3]
        pts_sonar = (self.scene_points @ R_sw.T) + t_sw

        # Filter by forward FOV (90 deg azimuth, 90 deg elevation, max range 12m)
        r = np.linalg.norm(pts_sonar, axis=1)
        az = np.arctan2(pts_sonar[:, 1], pts_sonar[:, 0])
        el = np.arctan2(pts_sonar[:, 2], pts_sonar[:, 0])

        in_fov = (pts_sonar[:, 0] > 0.3) & (r <= 12.0) & (np.abs(az) <= np.pi / 4.0) & (np.abs(el) <= np.pi / 4.0)
        scan = pts_sonar[in_fov]

        # Subsample to realistic point count (~250-400 points)
        if len(scan) > 300:
            idx = np.random.choice(len(scan), 300, replace=False)
            scan = scan[idx]

        # Add noise
        noise = np.random.normal(0, self.cfg.sonar_noise, scan.shape)
        scan += noise

        # Add occasional random scatter outliers
        if self.cfg.sonar_outlier_ratio > 0 and len(scan) > 0:
            n_out = int(len(scan) * self.cfg.sonar_outlier_ratio)
            outliers = np.random.uniform(0.5, 8.0, (n_out, 3))
            scan = np.vstack((scan, outliers))

        return scan.astype(np.float32)
