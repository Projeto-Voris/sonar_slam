"""ROS 2 message conversions and multi-type dispatchers."""
from typing import Tuple, Optional
import numpy as np
from std_msgs.msg import Header
from geometry_msgs.msg import (
    Point, Quaternion, Pose, PoseStamped, PoseWithCovarianceStamped,
    Twist, TwistWithCovarianceStamped, Vector3Stamped, TransformStamped
)
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2

from ..geometry import rot_to_quat, quat_to_rot, rot_to_euler


def numpy_to_pointcloud2(points: np.ndarray, stamp, frame_id: str = "base_link") -> PointCloud2:
    """Convert (N, 3) numpy float32/float64 array to sensor_msgs/PointCloud2."""
    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id

    if points is None or len(points) == 0:
        return pc2.create_cloud_xyz32(header, [])

    pts_f32 = points[:, :3].astype(np.float32)
    return pc2.create_cloud_xyz32(header, pts_f32)


def pointcloud2_to_numpy(msg: PointCloud2) -> np.ndarray:
    """Convert sensor_msgs/PointCloud2 to (N, 3) numpy array."""
    if msg.width == 0 or msg.height == 0:
        return np.empty((0, 3), dtype=np.float32)
    try:
        arr = pc2.read_points_numpy(msg, field_names=['x', 'y', 'z'])
        # Filter NaNs
        valid = ~np.isnan(arr).any(axis=1)
        return arr[valid].astype(np.float32)
    except Exception:
        gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        pts = list(gen)
        if len(pts) == 0:
            return np.empty((0, 3), dtype=np.float32)
        return np.array([[p[0], p[1], p[2]] for p in pts], dtype=np.float32)


def make_odometry_msg(p: np.ndarray, q: np.ndarray, v: np.ndarray,
                      cov6: Optional[np.ndarray], stamp,
                      frame_id: str = "odom", child_frame_id: str = "base_link") -> Odometry:
    """Construct nav_msgs/Odometry message."""
    msg = Odometry()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.child_frame_id = child_frame_id

    # Pose: q is [qw, qx, qy, qz]
    msg.pose.pose.position.x = float(p[0])
    msg.pose.pose.position.y = float(p[1])
    msg.pose.pose.position.z = float(p[2])
    msg.pose.pose.orientation.w = float(q[0])
    msg.pose.pose.orientation.x = float(q[1])
    msg.pose.pose.orientation.y = float(q[2])
    msg.pose.pose.orientation.z = float(q[3])

    if cov6 is not None:
        msg.pose.covariance = cov6.flatten().tolist()

    # Twist
    msg.twist.twist.linear.x = float(v[0])
    msg.twist.twist.linear.y = float(v[1])
    msg.twist.twist.linear.z = float(v[2])

    return msg


def make_pose_stamped(T: np.ndarray, stamp, frame_id: str = "map") -> PoseStamped:
    """Construct geometry_msgs/PoseStamped from 4x4 transform."""
    msg = PoseStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id

    q = rot_to_quat(T[:3, :3])
    msg.pose.position.x = float(T[0, 3])
    msg.pose.position.y = float(T[1, 3])
    msg.pose.position.z = float(T[2, 3])
    msg.pose.orientation.w = float(q[0])
    msg.pose.orientation.x = float(q[1])
    msg.pose.orientation.y = float(q[2])
    msg.pose.orientation.z = float(q[3])
    return msg


def make_transform_stamped(T: np.ndarray, stamp,
                           frame_id: str, child_frame_id: str) -> TransformStamped:
    """Construct geometry_msgs/TransformStamped for tf2 broadcasting."""
    msg = TransformStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.child_frame_id = child_frame_id

    q = rot_to_quat(T[:3, :3])
    msg.transform.translation.x = float(T[0, 3])
    msg.transform.translation.y = float(T[1, 3])
    msg.transform.translation.z = float(T[2, 3])
    msg.transform.rotation.w = float(q[0])
    msg.transform.rotation.x = float(q[1])
    msg.transform.rotation.y = float(q[2])
    msg.transform.rotation.z = float(q[3])
    return msg


def extract_depth(msg) -> Tuple[float, Optional[float]]:
    """Extract (depth_z_up, sigma) from message.
    Converts positive-down depth reading to Z-up: depth_z_up = -depth_down.
    """
    if hasattr(msg, 'pose') and hasattr(msg.pose, 'pose'):
        # PoseWithCovarianceStamped: position.z is depth positive down
        depth_down = msg.pose.pose.position.z
        cov = msg.pose.covariance[14] if hasattr(msg.pose, 'covariance') else 0.0
        sigma = np.sqrt(cov) if cov > 0 else None
        return -depth_down, sigma
    elif hasattr(msg, 'pose'):
        # PoseStamped
        depth_down = msg.pose.position.z
        return -depth_down, None
    elif hasattr(msg, 'data'):
        # std_msgs/Float64
        return -float(msg.data), None
    else:
        raise ValueError("Unsupported depth message")


def extract_twist(msg) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Extract (velocity_3d, 3x3 covariance) from DVL message."""
    if hasattr(msg, 'velocity'):
        v = msg.velocity
        cov = np.asarray(msg.velocity_covar, dtype=float).reshape(3, 3)
        valid = np.all(np.isfinite(cov)) and np.allclose(cov, cov.T) and np.min(np.linalg.eigvalsh(cov)) > 0
        return np.array([v.x, v.y, v.z], dtype=float), cov if valid else None
    if hasattr(msg, 'twist') and hasattr(msg.twist, 'twist'):
        # TwistWithCovarianceStamped
        tw = msg.twist.twist.linear
        vel = np.array([tw.x, tw.y, tw.z], dtype=np.float64)
        cov_full = np.array(msg.twist.covariance).reshape(6, 6)
        cov_3x3 = cov_full[0:3, 0:3]
        return vel, cov_3x3
    elif hasattr(msg, 'twist'):
        # TwistStamped
        tw = msg.twist.linear
        return np.array([tw.x, tw.y, tw.z], dtype=np.float64), None
    elif hasattr(msg, 'vector'):
        # Vector3Stamped
        v = msg.vector
        return np.array([v.x, v.y, v.z], dtype=np.float64), None
    return np.zeros(3), None


def extract_pose_and_yaw(msg) -> Tuple[np.ndarray, float, Optional[np.ndarray], Optional[float]]:
    """Extract (position_3d, yaw, pos_cov_3x3, yaw_cov) from visual pose message."""
    if hasattr(msg, 'pose') and hasattr(msg.pose, 'pose'):
        # PoseWithCovarianceStamped or Odometry
        p = msg.pose.pose.position
        pos = np.array([p.x, p.y, p.z], dtype=np.float64)
        o = msg.pose.pose.orientation
        q = np.array([o.w, o.x, o.y, o.z], dtype=np.float64)
        R = quat_to_rot(q)
        _, _, yaw = rot_to_euler(R)

        cov_full = np.array(msg.pose.covariance).reshape(6, 6)
        pos_cov = cov_full[0:3, 0:3]
        yaw_cov = cov_full[5, 5]
        return pos, yaw, pos_cov, yaw_cov
    elif hasattr(msg, 'pose'):
        # PoseStamped
        p = msg.pose.position
        pos = np.array([p.x, p.y, p.z], dtype=np.float64)
        o = msg.pose.orientation
        q = np.array([o.w, o.x, o.y, o.z], dtype=np.float64)
        R = quat_to_rot(q)
        _, _, yaw = rot_to_euler(R)
        return pos, yaw, None, None
    return np.zeros(3), 0.0, None, None
