"""State estimation ROS 2 node supporting ESKF and Complementary filter."""
import rclpy
from rclpy.node import Node
import numpy as np

from sensor_msgs.msg import Imu
from geometry_msgs.msg import TwistWithCovarianceStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster

from ..geometry import make_se3
from ..estimation.eskf import ESKF, EskfConfig
from ..estimation.complementary import ComplementaryFilter, ComplementaryConfig
from ..utils.conversions import (
    make_odometry_msg, make_transform_stamped,
    extract_depth, extract_twist, extract_pose_and_yaw
)
from ..utils.topics import sensor_qos, reliable_qos


class StateEstimationNode(Node):
    """Fuses IMU, DVL, Depth, and Stereo VO into a continuous dead-reckoning trajectory."""

    def __init__(self):
        super().__init__('state_estimation_node')

        # Parameters
        self.declare_parameter('method', 'eskf')  # 'eskf' or 'complementary'
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')

        # Sensor topic names
        self.declare_parameter('imu_topic', '/imu/data_raw')
        self.declare_parameter('dvl_topic', '/dvl/velocity')
        self.declare_parameter('depth_topic', '/altitude/depth')
        self.declare_parameter('stereo_topic', '/stereo/pose')

        # Sensor mounting offsets / lever arms (m, relative to base_link)
        self.declare_parameter('lever_arm_dvl', [0.1, 0.0, -0.2])
        self.declare_parameter('lever_arm_depth', [-0.2, 0.0, 0.1])
        self.declare_parameter('lever_arm_imu', [0.0, 0.0, 0.0])

        self.method = self.get_parameter('method').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value

        self.r_dvl = np.array(self.get_parameter('lever_arm_dvl').value, dtype=np.float64)
        self.r_depth = np.array(self.get_parameter('lever_arm_depth').value, dtype=np.float64)
        self.r_imu = np.array(self.get_parameter('lever_arm_imu').value, dtype=np.float64)

        # Filters
        if self.method == 'eskf':
            self.filter = ESKF(EskfConfig())
            self.get_logger().info('Initialized 15-state ESKF state estimator.')
        else:
            self.filter = ComplementaryFilter(ComplementaryConfig())
            self.get_logger().info('Initialized Complementary filter state estimator.')

        # Initialization buffer
        self.accel_init_samples = []
        self.is_initialized = False
        self.last_imu_stamp = None
        self.latest_gyro = np.zeros(3)

        # Broadcaster and Publisher
        self.tf_broadcaster = TransformBroadcaster(self)
        self.odom_pub = self.create_publisher(Odometry, '/state_estimation/odom', reliable_qos)

        # Subscriptions
        imu_topic = self.get_parameter('imu_topic').value
        dvl_topic = self.get_parameter('dvl_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        stereo_topic = self.get_parameter('stereo_topic').value

        self.sub_imu = self.create_subscription(Imu, imu_topic, self._on_imu, sensor_qos)
        self.sub_dvl = self.create_subscription(TwistWithCovarianceStamped, dvl_topic, self._on_dvl, sensor_qos)
        self.sub_depth = self.create_subscription(PoseWithCovarianceStamped, depth_topic, self._on_depth, sensor_qos)
        self.sub_stereo = self.create_subscription(PoseWithCovarianceStamped, stereo_topic, self._on_stereo, sensor_qos)

        self.get_logger().info('state_estimation_node is ready and listening for sensors.')

    def _on_imu(self, msg: Imu):
        gyro = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        accel = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        self.latest_gyro = gyro

        # 1. Level roll/pitch from initial static gravity readings (~20 samples)
        if not self.is_initialized:
            self.accel_init_samples.append(accel)
            if len(self.accel_init_samples) >= 20:
                self.filter.initialize_attitude(np.array(self.accel_init_samples))
                self.is_initialized = True
                self.get_logger().info('Initial attitude leveled successfully.')
            return

        # Calculate dt
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_imu_stamp is None:
            self.last_imu_stamp = stamp_sec
            return

        dt = stamp_sec - self.last_imu_stamp
        self.last_imu_stamp = stamp_sec

        if dt <= 0.0 or dt > 0.2:
            dt = 0.01

        # Predict
        if self.method == 'eskf':
            self.filter.predict(gyro, accel, dt)
        else:
            self.filter.update_imu(gyro, accel, dt)

        # Broadcast TF and publish odometry
        self._publish_state(msg.header.stamp)

    def _on_dvl(self, msg: TwistWithCovarianceStamped):
        if not self.is_initialized:
            return
        vel, cov = extract_twist(msg)
        R_bd = np.eye(3)  # DVL aligned with base_link axes

        if self.method == 'eskf':
            self.filter.correct_dvl(vel, R_bd, self.r_dvl, cov=cov, gyro=self.latest_gyro)
        else:
            self.filter.update_dvl(vel, R_bd, dt=0.1)

    def _on_depth(self, msg: PoseWithCovarianceStamped):
        if not self.is_initialized:
            return
        depth_z_up, sigma = extract_depth(msg)
        if self.method == 'eskf':
            self.filter.correct_depth(depth_z_up, self.r_depth, sigma=sigma)
        else:
            self.filter.update_depth(depth_z_up, self.r_depth)

    def _on_stereo(self, msg: PoseWithCovarianceStamped):
        if not self.is_initialized or self.method != 'eskf':
            return
        pos, yaw, pos_cov, yaw_cov = extract_pose_and_yaw(msg)
        self.filter.correct_visual_pose(pos, yaw, pos_cov, yaw_cov)

    def _publish_state(self, stamp):
        p = self.filter.p
        q = self.filter.q
        v = self.filter.v
        cov6 = self.filter.get_pose_covariance_6x6() if self.method == 'eskf' else None

        # 1. Publish Odometry
        odom_msg = make_odometry_msg(p, q, v, cov6, stamp, self.odom_frame, self.base_frame)
        self.odom_pub.publish(odom_msg)

        # 2. Broadcast TF odom -> base_link
        T_odom_base = make_se3(self.filter.R, p)
        tf_msg = make_transform_stamped(T_odom_base, stamp, self.odom_frame, self.base_frame)
        self.tf_broadcaster.sendTransform(tf_msg)


def main(args=None):
    rclpy.init(args=args)
    node = StateEstimationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
