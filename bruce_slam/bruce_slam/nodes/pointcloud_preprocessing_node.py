"""3-D Sonar point cloud preprocessing ROS 2 node."""
import rclpy
from rclpy.node import Node
import numpy as np

from sensor_msgs.msg import PointCloud2
from tf2_ros import Buffer, TransformListener

from ..geometry import make_se3, quat_to_rot
from ..perception.pointcloud_filter import PointCloudPreprocessor, FilterConfig
from ..utils.conversions import pointcloud2_to_numpy, numpy_to_pointcloud2
from ..utils.topics import sensor_qos, reliable_qos


class PointCloudPreprocessingNode(Node):
    """Crops, filters, voxel-downsamples 3-D sonar point clouds and expresses them in base_link."""

    def __init__(self):
        super().__init__('pointcloud_preprocessing_node')

        # Parameters
        self.declare_parameter('input_topic', '/sonar_3d/points')
        self.declare_parameter('output_topic', '/sonar_3d/points_filtered')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('sonar_frame', 'sonar_link')

        self.declare_parameter('min_range', 0.3)
        self.declare_parameter('max_range', 15.0)
        self.declare_parameter('fov_azimuth_deg', 90.0)
        self.declare_parameter('fov_elevation_deg', 90.0)
        self.declare_parameter('voxel_size', 0.15)
        self.declare_parameter('min_neighbors', 4)
        self.declare_parameter('outlier_radius', 0.4)

        # Extrinsics fallback (sonar mounted at x=0.4, y=0.0, z=-0.1 relative to base_link)
        self.declare_parameter('sonar_offset_xyz', [0.4, 0.0, -0.1])

        cfg = FilterConfig(
            min_range=self.get_parameter('min_range').value,
            max_range=self.get_parameter('max_range').value,
            fov_azimuth_deg=self.get_parameter('fov_azimuth_deg').value,
            fov_elevation_deg=self.get_parameter('fov_elevation_deg').value,
            voxel_size=self.get_parameter('voxel_size').value,
            min_neighbors=self.get_parameter('min_neighbors').value,
            outlier_radius=self.get_parameter('outlier_radius').value,
        )
        self.preprocessor = PointCloudPreprocessor(cfg)

        self.base_frame = self.get_parameter('base_frame').value
        self.sonar_frame = self.get_parameter('sonar_frame').value
        self.default_offset = np.array(self.get_parameter('sonar_offset_xyz').value, dtype=np.float64)

        # TF2 listener for dynamic/static extrinsics lookup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Sub / Pub
        in_topic = self.get_parameter('input_topic').value
        out_topic = self.get_parameter('output_topic').value
        self.sub_points = self.create_subscription(PointCloud2, in_topic, self._on_points, sensor_qos)
        self.pub_filtered = self.create_publisher(PointCloud2, out_topic, reliable_qos)

        self.get_logger().info('pointcloud_preprocessing_node is ready.')

    def _get_T_base_sonar(self) -> np.ndarray:
        """Query TF for sonar_link -> base_link transform."""
        try:
            tf_stamped = self.tf_buffer.lookup_transform(
                self.base_frame, self.sonar_frame, rclpy.time.Time()
            )
            t = tf_stamped.transform.translation
            r = tf_stamped.transform.rotation
            p = np.array([t.x, t.y, t.z], dtype=np.float64)
            q = np.array([r.w, r.x, r.y, r.z], dtype=np.float64)
            return make_se3(quat_to_rot(q), p)
        except Exception:
            # Fallback to configured offset
            return make_se3(np.eye(3), self.default_offset)

    def _on_points(self, msg: PointCloud2):
        pts = pointcloud2_to_numpy(msg)
        if len(pts) == 0:
            return

        T_base_sonar = self._get_T_base_sonar()
        pts_filtered = self.preprocessor.process(pts, T_base_sonar)

        # Publish filtered points expressed in base_link
        out_msg = numpy_to_pointcloud2(pts_filtered, msg.header.stamp, frame_id=self.base_frame)
        self.pub_filtered.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudPreprocessingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
