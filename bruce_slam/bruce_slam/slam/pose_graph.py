"""6-DoF GTSAM Pose3 / iSAM2 pose-graph SLAM backend."""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
import gtsam
from gtsam.symbol_shorthand import X

from ..geometry import make_se3, inv_se3, rot_to_quat, quat_to_rot, rot_to_euler, euler_to_rot
from .registration import (
    icp_point_to_plane_numpy,
    icp_point_to_plane_open3d,
    estimate_normals_pca,
    ICPResult,
    RegistrationConfig,
    HAS_OPEN3D
)


@dataclass
class SlamConfig:
    # Keyframe criteria
    keyframe_min_dt: float = 0.5          # Minimum time between keyframes (s)
    keyframe_translation: float = 0.5     # Translation threshold to spawn keyframe (m)
    keyframe_rotation_deg: float = 15.0   # Rotation threshold to spawn keyframe (deg)

    # Odometry noise [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]
    odom_sigmas: List[float] = field(default_factory=lambda: [0.02, 0.02, 0.03, 0.05, 0.05, 0.03])

    # SSM (Sequential Scan Matching)
    enable_ssm: bool = True
    ssm_min_fitness: float = 0.6
    ssm_max_translation: float = 1.0      # Gate against runaway sliding (m)
    ssm_max_rotation_deg: float = 20.0
    icp_backend: str = "numpy"            # "numpy" or "open3d"

    # NSSM (Loop closure)
    enable_loop_closure: bool = True
    loop_search_radius: float = 4.0       # Spatial radius for candidate search (m)
    loop_min_index_gap: int = 8           # Temporal separation (keyframes)
    loop_min_fitness: float = 0.75
    loop_max_candidates: int = 3
    loop_huber_k: float = 1.345

    # Visual SLAM factor
    enable_visual_factors: bool = True
    vis_trans_sigma: float = 0.08         # m
    vis_yaw_sigma: float = 0.03           # rad
    vis_roll_pitch_sigma: float = 1.0     # rad (heavily inflated: don't trust visual roll/pitch)

    # Map representation
    map_voxel_size: float = 0.1           # m


@dataclass
class Keyframe:
    index: int
    timestamp: float
    T_odom_base: np.ndarray       # 4x4 dead reckoning pose
    T_map_base: np.ndarray        # 4x4 SLAM optimized pose
    cloud: np.ndarray             # (N, 3) points in base_link frame
    normals: Optional[np.ndarray] = None


class PoseGraphSLAM:
    """Full 6-DoF GTSAM Pose3 / iSAM2 factor graph SLAM."""

    def __init__(self, config: SlamConfig = None):
        self.cfg = config or SlamConfig()
        self.reg_cfg = RegistrationConfig()

        # iSAM2 setup
        isam_params = gtsam.ISAM2Params()
        isam_params.setRelinearizeThreshold(0.1)
        isam_params.relinearizeSkip = 1
        self.isam = gtsam.ISAM2(isam_params)

        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_values = gtsam.Values()
        self.current_estimate = gtsam.Values()

        self.keyframes: List[Keyframe] = []
        self.last_keyframe_time = -1.0

        # Visual pose buffer for relative visual factors: {timestamp: (p_base, yaw_base)}
        self.visual_buffer: List[Tuple[float, np.ndarray, float]] = []

    @staticmethod
    def _to_gtsam_pose3(T: np.ndarray) -> gtsam.Pose3:
        """Convert 4x4 matrix to gtsam.Pose3."""
        R = gtsam.Rot3(T[:3, :3])
        t = gtsam.Point3(T[:3, 3])
        return gtsam.Pose3(R, t)

    @staticmethod
    def _from_gtsam_pose3(pose: gtsam.Pose3) -> np.ndarray:
        """Convert gtsam.Pose3 to 4x4 matrix."""
        return pose.matrix()

    def should_create_keyframe(self, t: float, T_odom_base: np.ndarray) -> bool:
        """Evaluate keyframe generation criteria."""
        if len(self.keyframes) == 0:
            return True

        dt = t - self.last_keyframe_time
        if dt < self.cfg.keyframe_min_dt:
            return False

        last_T = self.keyframes[-1].T_odom_base
        delta_T = inv_se3(last_T) @ T_odom_base
        d_trans = np.linalg.norm(delta_T[:3, 3])
        _, _, d_yaw = rot_to_euler(delta_T[:3, :3])
        d_rot = np.abs(d_yaw)

        rot_thresh = np.deg2rad(self.cfg.keyframe_rotation_deg)
        return (d_trans >= self.cfg.keyframe_translation) or (d_rot >= rot_thresh)

    def add_keyframe(self, t: float, T_odom_base: np.ndarray, cloud: np.ndarray) -> Keyframe:
        """Add new keyframe and update factor graph."""
        idx = len(self.keyframes)
        normals = estimate_normals_pca(cloud, k=15) if len(cloud) > 15 else None

        # Initial pose guess
        if idx == 0:
            T_map_base = T_odom_base.copy()
            # Prior factor on origin
            prior_sigmas = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
            prior_noise = gtsam.noiseModel.Diagonal.Sigmas(prior_sigmas)
            self.graph.add(gtsam.PriorFactorPose3(X(0), self._to_gtsam_pose3(T_map_base), prior_noise))
            self.initial_values.insert(X(0), self._to_gtsam_pose3(T_map_base))
        else:
            prev_kf = self.keyframes[-1]
            delta_odom = inv_se3(prev_kf.T_odom_base) @ T_odom_base
            T_map_base = prev_kf.T_map_base @ delta_odom

            self.initial_values.insert(X(idx), self._to_gtsam_pose3(T_map_base))

            # 1. Odometry BetweenFactor
            odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(self.cfg.odom_sigmas))
            self.graph.add(
                gtsam.BetweenFactorPose3(X(idx - 1), X(idx), self._to_gtsam_pose3(delta_odom), odom_noise)
            )

            # 2. Sequential Scan Matching (SSM)
            if self.cfg.enable_ssm and prev_kf.cloud is not None and len(prev_kf.cloud) > 20 and len(cloud) > 20:
                self._add_ssm_factor(idx, prev_kf, cloud, normals, delta_odom)

            # 3. Independent Relative Visual Factor
            if self.cfg.enable_visual_factors:
                self._maybe_add_visual_factor(idx, prev_kf.timestamp, t)

            # 4. Loop Closure Factor
            if self.cfg.enable_loop_closure and idx >= self.cfg.loop_min_index_gap:
                self._check_and_add_loop_closures(idx, cloud, normals, T_map_base)

        # Optimize graph incrementally with iSAM2
        self.isam.update(self.graph, self.initial_values)
        self.current_estimate = self.isam.calculateEstimate()

        # Clear step factor buffers
        self.graph.resize(0)
        self.initial_values.clear()

        # Update optimized pose for all keyframes
        for kf in self.keyframes:
            if self.current_estimate.exists(X(kf.index)):
                kf.T_map_base = self._from_gtsam_pose3(self.current_estimate.atPose3(X(kf.index)))

        opt_pose = self._from_gtsam_pose3(self.current_estimate.atPose3(X(idx)))
        kf = Keyframe(
            index=idx,
            timestamp=t,
            T_odom_base=T_odom_base.copy(),
            T_map_base=opt_pose,
            cloud=cloud.copy(),
            normals=normals
        )
        self.keyframes.append(kf)
        self.last_keyframe_time = t
        return kf

    def _add_ssm_factor(self, idx: int, prev_kf: Keyframe, cloud: np.ndarray,
                        normals: np.ndarray, delta_odom: np.ndarray):
        """Perform sequential scan matching between consecutive keyframes."""
        if self.cfg.icp_backend == "open3d" and HAS_OPEN3D:
            icp_res = icp_point_to_plane_open3d(
                source_pts=cloud,
                target_pts=prev_kf.cloud,
                target_normals=prev_kf.normals,
                init_T=delta_odom,
                config=self.reg_cfg
            )
        else:
            icp_res = icp_point_to_plane_numpy(
                source_pts=cloud,
                target_pts=prev_kf.cloud,
                target_normals=prev_kf.normals,
                init_T=delta_odom,
                config=self.reg_cfg
            )

        # Gate ICP result against odometry divergence
        T_diff = inv_se3(delta_odom) @ icp_res.T
        trans_diff = np.linalg.norm(T_diff[:3, 3])
        _, _, yaw_diff = rot_to_euler(T_diff[:3, :3])
        rot_diff = np.abs(yaw_diff)

        if (icp_res.fitness >= self.cfg.ssm_min_fitness and
            trans_diff <= self.cfg.ssm_max_translation and
            rot_diff <= np.deg2rad(self.cfg.ssm_max_rotation_deg)):

            # Adaptive covariance from ICP Hessian
            cov_6x6 = icp_res.covariance.copy()
            # Enforce positive definiteness and symmetry
            cov_6x6 = 0.5 * (cov_6x6 + cov_6x6.T)
            min_eig = np.min(np.real(np.linalg.eigvals(cov_6x6)))
            if min_eig < 1e-6:
                cov_6x6 += np.eye(6) * (1e-6 - min_eig)

            icp_noise = gtsam.noiseModel.Gaussian.Covariance(cov_6x6)
            self.graph.add(
                gtsam.BetweenFactorPose3(
                    X(idx - 1), X(idx), self._to_gtsam_pose3(icp_res.T), icp_noise
                )
            )

    def _maybe_add_visual_factor(self, idx: int, t_prev: float, t_curr: float):
        """Add independent relative visual odometry factor between keyframe timestamps."""
        # Find visual poses closest to t_prev and t_curr
        if len(self.visual_buffer) < 2:
            return

        def find_closest(target_t):
            closest = min(self.visual_buffer, key=lambda item: abs(item[0] - target_t))
            if abs(closest[0] - target_t) < 0.2:
                return closest
            return None

        v_prev = find_closest(t_prev)
        v_curr = find_closest(t_curr)

        if v_prev is not None and v_curr is not None:
            # Relative visual transform in body frame
            p_prev, yaw_prev = v_prev[1], v_prev[2]
            p_curr, yaw_curr = v_curr[1], v_curr[2]

            R_prev = euler_to_rot(0.0, 0.0, yaw_prev)
            T_prev = make_se3(R_prev, p_prev)

            R_curr = euler_to_rot(0.0, 0.0, yaw_curr)
            T_curr = make_se3(R_curr, p_curr)

            delta_vis = inv_se3(T_prev) @ T_curr

            # Noise model: high uncertainty on roll and pitch, tight on x,y,z,yaw
            sigmas = np.array([
                self.cfg.vis_roll_pitch_sigma,
                self.cfg.vis_roll_pitch_sigma,
                self.cfg.vis_yaw_sigma,
                self.cfg.vis_trans_sigma,
                self.cfg.vis_trans_sigma,
                self.cfg.vis_trans_sigma
            ])
            vis_noise = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
            self.graph.add(
                gtsam.BetweenFactorPose3(
                    X(idx - 1), X(idx), self._to_gtsam_pose3(delta_vis), vis_noise
                )
            )

    def _check_and_add_loop_closures(self, idx: int, cloud: np.ndarray,
                                     normals: np.ndarray, T_curr_est: np.ndarray):
        """Search past keyframes for spatial loop closure."""
        candidates = []
        curr_pos = T_curr_est[:3, 3]

        for kf in self.keyframes[:-self.cfg.loop_min_index_gap]:
            dist = np.linalg.norm(kf.T_map_base[:3, 3] - curr_pos)
            if dist < self.cfg.loop_search_radius:
                candidates.append((dist, kf))

        candidates.sort(key=lambda x: x[0])
        for _, kf_cand in candidates[:self.cfg.loop_max_candidates]:
            # Initial guess: relative pose in map frame
            init_T = inv_se3(kf_cand.T_map_base) @ T_curr_est

            icp_res = icp_point_to_plane_numpy(
                source_pts=cloud,
                target_pts=kf_cand.cloud,
                target_normals=kf_cand.normals,
                init_T=init_T,
                config=self.reg_cfg
            )

            if icp_res.fitness >= self.cfg.loop_min_fitness:
                # Add robust loop factor
                cov_6x6 = icp_res.covariance.copy() + np.eye(6) * 1e-4
                base_noise = gtsam.noiseModel.Gaussian.Covariance(cov_6x6)
                robust_noise = gtsam.noiseModel.Robust.Create(
                    gtsam.noiseModel.mEstimator.Huber.Create(self.cfg.loop_huber_k),
                    base_noise
                )
                self.graph.add(
                    gtsam.BetweenFactorPose3(
                        X(kf_cand.index), X(idx), self._to_gtsam_pose3(icp_res.T), robust_noise
                    )
                )

    def get_map_to_odom_transform(self) -> np.ndarray:
        """Return 6-DoF correction transform T_map_odom = T_map_base * T_odom_base^-1."""
        if len(self.keyframes) == 0:
            return np.eye(4, dtype=np.float64)
        latest_kf = self.keyframes[-1]
        return latest_kf.T_map_base @ inv_se3(latest_kf.T_odom_base)

    def get_accumulated_map(self) -> np.ndarray:
        """Transform all keyframe point clouds into map frame and return merged point cloud."""
        if len(self.keyframes) == 0:
            return np.empty((0, 3), dtype=np.float32)

        map_pts = []
        for kf in self.keyframes:
            if kf.cloud is not None and len(kf.cloud) > 0:
                R = kf.T_map_base[:3, :3]
                t = kf.T_map_base[:3, 3]
                pts_map = (kf.cloud @ R.T) + t
                map_pts.append(pts_map)

        if len(map_pts) == 0:
            return np.empty((0, 3), dtype=np.float32)

        merged = np.vstack(map_pts)
        if self.cfg.map_voxel_size > 0.0:
            coords = np.floor(merged / self.cfg.map_voxel_size).astype(np.int64)
            _, idxs = np.unique(coords, axis=0, return_index=True)
            merged = merged[idxs]
        return merged.astype(np.float32)
