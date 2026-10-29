"""3-D Sonar point cloud preprocessing: crop, voxel grid, outlier filter."""
from dataclasses import dataclass
import numpy as np


@dataclass
class FilterConfig:
    min_range: float = 0.3          # Minimum range (blind zone crop, m)
    max_range: float = 15.0         # Maximum range (m)
    fov_azimuth_deg: float = 90.0   # Horizontal FOV (deg)
    fov_elevation_deg: float = 90.0 # Vertical FOV (deg)
    voxel_size: float = 0.15        # Downsampling voxel grid resolution (m)
    min_neighbors: int = 4          # Statistical / radius outlier removal min neighbors
    outlier_radius: float = 0.4     # Neighborhood radius for outlier filter (m)


class PointCloudPreprocessor:
    """ROS-free preprocessing pipeline for 3-D forward-looking sonar point clouds."""

    def __init__(self, config: FilterConfig = None):
        self.cfg = config or FilterConfig()
        self.half_azimuth = np.deg2rad(self.cfg.fov_azimuth_deg * 0.5)
        self.half_elevation = np.deg2rad(self.cfg.fov_elevation_deg * 0.5)

    def process(self, points: np.ndarray, T_base_sonar: np.ndarray = None) -> np.ndarray:
        """Run crop, transform to base_link, voxel downsample, and outlier filter.
        Args:
            points: (N, 3) point cloud in sonar_link frame.
            T_base_sonar: 4x4 transform from sonar_link to base_link.
        Returns:
            (M, 3) filtered point cloud in base_link frame.
        """
        if points is None or len(points) == 0:
            return np.empty((0, 3), dtype=np.float32)

        # 1. Range & FOV crop in sonar frame (X forward, Y left, Z up)
        r = np.linalg.norm(points, axis=1)
        valid = (r >= self.cfg.min_range) & (r <= self.cfg.max_range)

        # FOV check
        azimuth = np.arctan2(points[:, 1], points[:, 0])
        elevation = np.arctan2(points[:, 2], points[:, 0])
        valid &= (np.abs(azimuth) <= self.half_azimuth) & (np.abs(elevation) <= self.half_elevation)
        pts = points[valid]

        if len(pts) == 0:
            return np.empty((0, 3), dtype=np.float32)

        # 2. Transform into base_link frame if transform provided
        if T_base_sonar is not None:
            R = T_base_sonar[:3, :3]
            t = T_base_sonar[:3, 3]
            pts = (pts @ R.T) + t

        # 3. Voxel downsample
        if self.cfg.voxel_size > 0.0:
            pts = self.voxel_downsample(pts, self.cfg.voxel_size)

        # 4. Outlier removal
        if self.cfg.min_neighbors > 0 and len(pts) > self.cfg.min_neighbors:
            pts = self.radius_outlier_removal(pts, self.cfg.outlier_radius, self.cfg.min_neighbors)

        return pts.astype(np.float32)

    @staticmethod
    def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
        """Voxel downsampling using hash table grouping in pure NumPy."""
        if len(points) == 0:
            return points
        coords = np.floor(points / voxel_size).astype(np.int64)
        # Unique voxel coordinates
        _, indices = np.unique(coords, axis=0, return_index=True)
        return points[indices]

    @staticmethod
    def radius_outlier_removal(points: np.ndarray, radius: float, min_neighbors: int) -> np.ndarray:
        """Simple, robust radius outlier filter."""
        from scipy.spatial import cKDTree
        if len(points) <= min_neighbors:
            return points
        tree = cKDTree(points)
        # Count neighbors within radius
        counts = tree.query_ball_point(points, r=radius, return_sorted=False)
        keep = np.array([len(c) - 1 >= min_neighbors for c in counts])
        return points[keep]
