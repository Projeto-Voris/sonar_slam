"""SLAM module: Point-to-plane ICP registration and GTSAM Pose3 pose graph."""
from .registration import (
    icp_point_to_plane_numpy,
    icp_point_to_plane_open3d,
    estimate_normals_pca,
    ICPResult,
    RegistrationConfig,
    HAS_OPEN3D
)
from .pose_graph import PoseGraphSLAM, SlamConfig

__all__ = [
    "icp_point_to_plane_numpy",
    "icp_point_to_plane_open3d",
    "estimate_normals_pca",
    "ICPResult",
    "RegistrationConfig",
    "HAS_OPEN3D",
    "PoseGraphSLAM",
    "SlamConfig"
]
