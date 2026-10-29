"""Full 6-DoF 3-D Sonar SLAM ROS 2 node."""
import rclpy
from rclpy.node import Node
import numpy as np

from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from tf2_ros import TransformBroadcaster

from ..geometry import make_se3, quat_to_rot, rot_to_quat, interpolate_pose
from ..slam.pose_graph import PoseGraphSLAM, SlamConfig
from ..utils.conversions import (
    pointcloud2_to_numpy, numpy_to_pointcloud2,
    make_odometry_msg, make_pose_stamped, make_transform_stamped,
    extract_pose_and_yaw
)
from ..utils.topics import sensor_qos, reliable_qos, latched_qos


class SlamNode(Node):
    """iSAM2 Pose3 3-D Sonar SLAM node with keyframe management, 3-D ICP, and map publishing."""

    def __init__(self):
        super().__init__('slam_node')

        # Parameters
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')

        self.declare_parameter('keyframe_min_dt', 0.5)
        self.declare_parameter('keyframe_translation', 0.5)
        self.declare_parameter('keyframe_rotation_deg', 15.0)

        self.declare_parameter('enable_ssm', True)
        self.declare_parameter('ssm_min_fitness', 0.6)
        self.declare_parameter('ssm_max_translation', 1.0)
        self.declare_parameter('icp_backend', 'numpy')  # 'numpy' or 'open3d'

        self.declare_parameter('enable_loop_closure', True)
        self.declare_parameter('loop_search_radius', 4.0)
        self.declare_parameter('loop_min_index_gap', 8)
        self.declare_parameter('loop_min_fitness', 0.75)

        self.declare_parameter('map_publish_period', 2.0)
        self.declare_parameter('odom_interpolation_max_gap', 0.2)

        self.map_frame = self.get_parameter('map_frame').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value

        # Initialize SLAM core
        cfg = SlamConfig(
            keyframe_min_dt=self.get_parameter('keyframe_min_dt').value,
            keyframe_translation=self.get_parameter('keyframe_translation').value,
            keyframe_rotation_deg=self.get_parameter('keyframe_rotation_deg').value,
            enable_ssm=self.get_parameter('enable_ssm').value,
            ssm_min_fitness=self.get_parameter('ssm_min_fitness').value,
            ssm_max_translation=self.get_parameter('ssm_max_translation').value,
            icp_backend=self.get_parameter('icp_backend').value,
            enable_loop_closure=self.get_parameter('enable_loop_closure').value,
            loop_search_radius=self.get_parameter('loop_search_radius').value,
            loop_min_index_gap=self.get_parameter('loop_min_index_gap').value,
            loop_min_fitness=self.get_parameter('loop_min_fitness').value,
        )
        self.slam = PoseGraphSLAM(cfg)

        # Odometry buffer: list of (t, T_odom_base, v)
        self.odom_buffer = []
        self.latest_odom_T = np.eye(4)
        self.latest_odom_v = np.zeros(3)

        # Broadcaster and Publishers
        self.tf_broadcaster = TransformBroadcaster(self)
        self.pub_slam_odom = self.create_publisher(Odometry, '/slam/odom', reliable_qos)
        self.pub_slam_pose = self.create_publisher(PoseStamped, '/slam/pose', reliable_qos)
        self.pub_path = self.create_publisher(Path, '/slam/path', latched_qos)
        self.pub_map = self.create_publisher(PointCloud2, '/slam/map_points', latched_qos)

        self.path_msg = Path()
        self.path_msg.header.frame_id = self.map_frame

        # Subscriptions
        self.sub_points = self.create_subscription(
            PointCloud2, '/sonar_3d/points_filtered', self._on_points, reliable_qos
        )
        self.sub_odom = self.create_subscription(
            Odometry, '/state_estimation/odom', self._on_odom, reliable_qos
        )
        self.sub_stereo = self.create_subscription(
            PoseWithCovarianceStamped, '/stereo/pose', self._on_stereo, sensor_qos
        )

        # Map publication timer
        map_period = self.get_parameter('map_publish_period').value
        self.map_timer = self.create_timer(map_period, self._publish_map)

        self.get_logger().info('slam_node is ready.')

    def _on_odom(self, msg: Odometry):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        pos = np.array([p.x, p.y, p.z], dtype=np.float64)
        q = np.array([o.w, o.x, o.y, o.z], dtype=np.float64)
        v = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])

        T = make_se3(quat_to_rot(q), pos)
        self.latest_odom_T = T
        self.latest_odom_v = v

        self.odom_buffer.append((t, T, v))
        # Keep buffer bounded (last 10 seconds)
        if len(self.odom_buffer) > 200:
            self.odom_buffer.pop(0)

        # Continuously broadcast current map -> odom TF
        T_map_odom = self.slam.get_map_to_odom_transform()
        tf_msg = make_transform_stamped(T_map_odom, msg.header.stamp, self.map_frame, self.odom_frame)
        self.tf_broadcaster.sendTransform(tf_msg)

    def _on_stereo(self, msg: PoseWithCovarianceStamped):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        pos, yaw, _, _ = extract_pose_and_yaw(msg)
        self.slam.visual_buffer.append((t, pos, yaw))
        if len(self.slam.visual_buffer) > 100:
            self.slam.visual_buffer.pop(0)

    def _on_points(self, msg: PointCloud2):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        pts = pointcloud2_to_numpy(msg)

        if len(pts) < 15:
            return

        T_odom_base = interpolate_pose(
            self.odom_buffer, t, self.get_parameter('odom_interpolation_max_gap').value
        )
        if T_odom_base is None:
            return

        # Check keyframe criteria
        if self.slam.should_create_keyframe(t, T_odom_base):
            kf = self.slam.add_keyframe(t, T_odom_base, pts)
            self.get_logger().info(f'Added Keyframe #{kf.index} at t={t:.2f}s (Total keyframes: {len(self.slam.keyframes)})')

            # Broadcast updated map -> odom TF
            T_map_odom = self.slam.get_map_to_odom_transform()
            tf_msg = make_transform_stamped(T_map_odom, msg.header.stamp, self.map_frame, self.odom_frame)
            self.tf_broadcaster.sendTransform(tf_msg)

            # Publish SLAM Pose & Path
            T_map_base = kf.T_map_base
            pose_stamped = make_pose_stamped(T_map_base, msg.header.stamp, self.map_frame)
            self.pub_slam_pose.publish(pose_stamped)

            self.path_msg.header.stamp = msg.header.stamp
            self.path_msg.poses.append(pose_stamped)
            self.pub_path.publish(self.path_msg)

            # Publish SLAM Odometry
            q_map = rot_to_quat(T_map_base[:3, :3])
            slam_odom = make_odometry_msg(
                T_map_base[:3, 3], q_map, self.latest_odom_v, None,
                msg.header.stamp, self.map_frame, self.base_frame
            )
            self.pub_slam_odom.publish(slam_odom)

    def _publish_map(self):
        """Periodically broadcast accumulated keyframe point cloud."""
        map_pts = self.slam.get_accumulated_map()
        if len(map_pts) > 0:
            cloud_msg = numpy_to_pointcloud2(map_pts, self.get_clock().now().to_msg(), frame_id=self.map_frame)
            self.pub_map.publish(cloud_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SlamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
