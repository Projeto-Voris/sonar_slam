"""Point-to-plane ICP registration with Hessian-based adaptive covariance and Open3D support."""
from dataclasses import dataclass
import numpy as np
import scipy.spatial
import scipy.linalg

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

from ..geometry import exp_so3, make_se3, inv_se3


@dataclass
class RegistrationConfig:
    max_iterations: int = 40
    tolerance: float = 1e-5
    k_neighbors_normal: int = 15
    max_corr_dist: float = 0.8
    mad_multiplier: float = 3.0
    icp_sigma_floor_trans: float = 0.05   # Minimum translation sigma (m)
    icp_sigma_floor_rot: float = 0.02     # Minimum rotation sigma (rad)
    anchor_weight: float = 1e2            # Prior odometry anchor weight for degenerate directions


@dataclass
class ICPResult:
    T: np.ndarray              # 4x4 relative transformation (source -> target)
    fitness: float             # Inlier ratio (0 to 1)
    rmse: float                # Root mean square residual
    covariance: np.ndarray     # 6x6 covariance [rot(3), trans(3)] (tangent order)
    converged: bool


def estimate_normals_pca(points: np.ndarray, k: int = 15) -> np.ndarray:
    """Estimate surface normals for a 3-D point cloud using PCA on local k-NN neighborhoods."""
    n_pts = len(points)
    if n_pts < k:
        normals = np.zeros_like(points)
        normals[:, 0] = -1.0
        return normals

    tree = scipy.spatial.cKDTree(points)
    _, idxs = tree.query(points, k=k)

    normals = np.zeros_like(points)
    for i in range(n_pts):
        neighbors = points[idxs[i]]
        centroid = np.mean(neighbors, axis=0)
        cov = (neighbors - centroid).T @ (neighbors - centroid)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]  # Smallest eigenvalue corresponds to normal

        # Orient towards origin / sensor
        if np.dot(normal, points[i]) > 0:
            normal = -normal
        normals[i] = normal

    return normals


def icp_point_to_plane_numpy(
    source_pts: np.ndarray,
    target_pts: np.ndarray,
    target_normals: np.ndarray = None,
    init_T: np.ndarray = None,
    config: RegistrationConfig = None
) -> ICPResult:
    """NumPy point-to-plane ICP with Hessian-based adaptive covariance."""
    cfg = config or RegistrationConfig()
    T = init_T.copy() if init_T is not None else np.eye(4, dtype=np.float64)

    if len(source_pts) < 15 or len(target_pts) < 15:
        return ICPResult(T, 0.0, 999.0, np.eye(6) * 1e6, False)

    if target_normals is None:
        target_normals = estimate_normals_pca(target_pts, k=cfg.k_neighbors_normal)

    target_tree = scipy.spatial.cKDTree(target_pts)
    prev_rmse = float("inf")
    converged = False

    # Floor information matrix (regularizer)
    sigmas = np.array([
        cfg.icp_sigma_floor_rot, cfg.icp_sigma_floor_rot, cfg.icp_sigma_floor_rot,
        cfg.icp_sigma_floor_trans, cfg.icp_sigma_floor_trans, cfg.icp_sigma_floor_trans
    ])
    Lambda_floor = np.diag(1.0 / (sigmas**2))

    H = np.copy(Lambda_floor)
    residuals = np.array([1.0])

    for it in range(cfg.max_iterations):
        # Transform source points with current T
        R = T[:3, :3]
        t = T[:3, 3]
        src_trans = (source_pts @ R.T) + t

        # Find closest points in target
        dists, idxs = target_tree.query(src_trans)

        # Robust correspondence rejection (Median + k * MAD)
        med = np.median(dists)
        mad = np.median(np.abs(dists - med))
        gate = min(cfg.max_corr_dist, med + cfg.mad_multiplier * max(mad, 1e-4))
        valid = dists <= gate

        n_valid = np.sum(valid)
        if n_valid < 10:
            break

        p_src = src_trans[valid]
        q_tgt = target_pts[idxs[valid]]
        n_tgt = target_normals[idxs[valid]]

        # Point-to-plane residual: r_i = n_i^T (q_i - p_i)
        # Note: in Gauss-Newton step, delta_x = [alpha, delta_t]
        # p_new ~= p + alpha x p + delta_t
        # residual = n^T (q - p)
        # Jacobian J_i = [p x n, n]
        cross_pn = np.cross(p_src, n_tgt)
        J = np.hstack((cross_pn, n_tgt))  # (N, 6): rot, trans
        residuals = np.sum((q_tgt - p_src) * n_tgt, axis=1)

        # Gauss-Newton step
        H = J.T @ J
        g = J.T @ residuals

        # Damping to prevent divergence
        H_damped = H + Lambda_floor * 1e-2
        try:
            delta = np.linalg.solve(H_damped, g)
        except np.linalg.LinAlgError:
            break

        # Apply update: dT = [Exp(delta_rot), delta_t]
        dR = exp_so3(delta[:3])
        dt_vec = delta[3:]
        dT = make_se3(dR, dt_vec)
        T = dT @ T

        rmse = np.sqrt(np.mean(residuals**2))
        if np.abs(prev_rmse - rmse) < cfg.tolerance:
            converged = True
            break
        prev_rmse = rmse

    fitness = float(np.sum(valid)) / float(len(source_pts))
    rmse = float(np.sqrt(np.mean(residuals**2)))

    # Compute adaptive covariance from Hessian: Cov = (H + eps*I)^-1 + sigma_floor^2
    # Tangent order: [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]
    try:
        eps_unobs = 1e-4
        cov_raw = np.linalg.inv(H + np.eye(6) * eps_unobs)
        cov = cov_raw + np.diag(sigmas**2)
    except np.linalg.LinAlgError:
        cov = np.eye(6) * 1e6

    return ICPResult(T, fitness, rmse, cov, converged)


def icp_point_to_plane_open3d(
    source_pts: np.ndarray,
    target_pts: np.ndarray,
    target_normals: np.ndarray = None,
    init_T: np.ndarray = None,
    config: RegistrationConfig = None
) -> ICPResult:
    """Open3D point-to-plane ICP implementation with information matrix extraction."""
    if not HAS_OPEN3D:
        # Fallback to NumPy
        return icp_point_to_plane_numpy(source_pts, target_pts, target_normals, init_T, config)

    cfg = config or RegistrationConfig()
    init_mat = init_T if init_T is not None else np.eye(4, dtype=np.float64)

    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(source_pts)

    pcd_tgt = o3d.geometry.PointCloud()
    pcd_tgt.points = o3d.utility.Vector3dVector(target_pts)
    if target_normals is not None:
        pcd_tgt.normals = o3d.utility.Vector3dVector(target_normals)
    else:
        pcd_tgt.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=cfg.k_neighbors_normal)
        )

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=cfg.max_iterations,
        relative_fitness=cfg.tolerance,
        relative_rmse=cfg.tolerance
    )

    reg = o3d.pipelines.registration.registration_icp(
        pcd_src,
        pcd_tgt,
        max_correspondence_distance=cfg.max_corr_dist,
        init=init_mat,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=criteria
    )

    # Information matrix to covariance
    info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        pcd_src, pcd_tgt, cfg.max_corr_dist, reg.transformation
    )

    sigmas = np.array([
        cfg.icp_sigma_floor_rot, cfg.icp_sigma_floor_rot, cfg.icp_sigma_floor_rot,
        cfg.icp_sigma_floor_trans, cfg.icp_sigma_floor_trans, cfg.icp_sigma_floor_trans
    ])
    Lambda_floor = np.diag(1.0 / (sigmas**2))

    try:
        cov = np.linalg.inv(info + Lambda_floor * 1e-3) + np.diag(sigmas**2)
    except np.linalg.LinAlgError:
        cov = np.eye(6) * 1e6

    converged = reg.fitness > 0.3
    return ICPResult(reg.transformation, float(reg.fitness), float(reg.inlier_rmse), cov, converged)
