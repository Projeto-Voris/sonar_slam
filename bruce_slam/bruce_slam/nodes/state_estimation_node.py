"""State estimation ROS 2 node supporting ESKF and Complementary filter."""
import rclpy
from rclpy.node import Node
import numpy as np
from dataclasses import fields
from rosidl_runtime_py.utilities import get_message

from sensor_msgs.msg import Imu
from geometry_msgs.msg import TwistWithCovarianceStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster, Buffer, TransformListener, TransformException

from ..geometry import make_se3, quat_to_rot
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

        for config_type, prefix in [(EskfConfig, 'eskf'), (ComplementaryConfig, 'complementary')]:
            defaults = config_type()
            for field in fields(defaults):
                self.declare_parameter(f'{prefix}.{field.name}', getattr(defaults, field.name))
        for name, default in {
            'dvl_message_type': 'geometry_msgs/msg/TwistWithCovarianceStamped',
            'depth_message_type': 'geometry_msgs/msg/PoseWithCovarianceStamped',
            'dvl_frame': '', 'depth_frame': '', 'imu_frame': '', 'use_tf_extrinsics': True,
            'pressure_surface_pa': 101325.0, 'water_density': 1025.0,
            'imu_max_gap': 0.5, 'sensor_max_age': 0.3,
            'initialization_samples': 20, 'initialization_gyro_threshold': 0.03,
            'initialization_accel_threshold': 0.2,
        }.items():
            self.declare_parameter(name, default)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.depth_initialized = False
        self.last_dvl_stamp = None

        # Filters
        if self.method == 'eskf':
            self.filter = ESKF(EskfConfig(**{f.name: self.get_parameter(f'eskf.{f.name}').value for f in fields(EskfConfig)}))
            self.get_logger().info('Initialized 15-state ESKF state estimator.')
        else:
            self.filter = ComplementaryFilter(ComplementaryConfig(**{f.name: self.get_parameter(f'complementary.{f.name}').value for f in fields(ComplementaryConfig)}))
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
        self.sub_dvl = self.create_subscription(get_message(self.get_parameter('dvl_message_type').value), dvl_topic, self._on_dvl, sensor_qos)
        self.sub_depth = self.create_subscription(get_message(self.get_parameter('depth_message_type').value), depth_topic, self._on_depth, sensor_qos)
        self.sub_stereo = self.create_subscription(PoseWithCovarianceStamped, stereo_topic, self._on_stereo, sensor_qos)

        self.get_logger().info('state_estimation_node is ready and listening for sensors.')

    def _on_imu(self, msg: Imu):
        gyro = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        accel = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        if not (np.all(np.isfinite(gyro)) and np.all(np.isfinite(accel))):
            return
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_imu_stamp is not None and stamp_sec <= self.last_imu_stamp:
            return
        self.latest_gyro = gyro

        extrinsics = self._extrinsics(msg, 'imu_frame', self.r_imu)
        if extrinsics is None:
            return
        R_bi, _ = extrinsics
        gyro, accel = R_bi @ gyro, R_bi @ accel
        self.latest_gyro = gyro

        # 1. Level roll/pitch from initial static gravity readings (~20 samples)
        if not self.is_initialized:
            if (np.linalg.norm(gyro) >= self.get_parameter('initialization_gyro_threshold').value or
                    abs(np.linalg.norm(accel) - self.filter.cfg.gravity) >= self.get_parameter('initialization_accel_threshold').value):
                self.accel_init_samples.clear()
                return
            self.accel_init_samples.append(accel)
            if len(self.accel_init_samples) >= self.get_parameter('initialization_samples').value:
                self.filter.initialize_attitude(np.array(self.accel_init_samples))
                self.is_initialized = True
                self.last_imu_stamp = stamp_sec
                self.get_logger().info('Initial attitude leveled successfully.')
            return

        # Calculate dt
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_imu_stamp is None:
            self.last_imu_stamp = stamp_sec
            return

        dt = stamp_sec - self.last_imu_stamp
        self.last_imu_stamp = stamp_sec

        if dt > min(0.5, self.get_parameter('imu_max_gap').value):
            if self.method == 'eskf':
                self.filter.dvl_stationary = False
                self.filter.stationary_duration = 0.0
            self.get_logger().warning('IMU gap exceeds limit; skipping propagation.')
            return

        # Predict
        if self.method == 'eskf':
            self.filter.predict(gyro, accel, dt)
        else:
            self.filter.update_imu(gyro, accel, dt)

        # Broadcast TF and publish odometry
        self._publish_state(msg.header.stamp)

    def _extrinsics(self, msg, frame_parameter, fallback):
        frame = self.get_parameter(frame_parameter).value or getattr(msg.header, 'frame_id', '')
        if frame == self.base_frame:
            return np.eye(3), np.zeros(3)
        if not self.get_parameter('use_tf_extrinsics').value or not frame:
            return np.eye(3), fallback
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, frame, rclpy.time.Time.from_msg(msg.header.stamp))
        except TransformException:
            self.get_logger().warning(f'Waiting for extrinsics from {frame} to {self.base_frame}.', throttle_duration_sec=5.0)
            return None
        r, t = tf.transform.rotation, tf.transform.translation
        return quat_to_rot(np.array([r.w, r.x, r.y, r.z])), np.array([t.x, t.y, t.z])

    def _sensor_time(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_imu_stamp is None or abs(stamp - self.last_imu_stamp) > self.get_parameter('sensor_max_age').value:
            return None
        return stamp

    def _on_dvl(self, msg: TwistWithCovarianceStamped):
        if not self.is_initialized:
            return
        stamp = self._sensor_time(msg)
        if stamp is None or (self.last_dvl_stamp is not None and stamp <= self.last_dvl_stamp):
            return
        if hasattr(msg, 'velocity_mode') and msg.velocity_mode != msg.DVL_MODE_BOTTOM:
            if self.method == 'eskf':
                self.filter.dvl_stationary = False
            return
        extrinsics = self._extrinsics(msg, 'dvl_frame', self.r_dvl)
        if extrinsics is None:
            return
        R_bd, r_dvl = extrinsics
        vel, cov = extract_twist(msg)

        if self.method == 'eskf':
            self.filter.correct_dvl(vel, R_bd, r_dvl, cov=cov, gyro=self.latest_gyro)
        else:
            dt = stamp - self.last_dvl_stamp if self.last_dvl_stamp is not None else 0.0
            vel = vel - R_bd.T @ np.cross(self.latest_gyro - self.filter.bg, r_dvl)
            self.filter.update_dvl(vel, R_bd, dt=dt)
        self.last_dvl_stamp = stamp

    def _on_depth(self, msg: PoseWithCovarianceStamped):
        if not self.is_initialized:
            return
        if self._sensor_time(msg) is None:
            return
        extrinsics = self._extrinsics(msg, 'depth_frame', self.r_depth)
        if extrinsics is None:
            return
        _, r_depth = extrinsics
        if hasattr(msg, 'fluid_pressure'):
            scale = self.get_parameter('water_density').value * self.filter.cfg.gravity
            depth_z_up = -(msg.fluid_pressure - self.get_parameter('pressure_surface_pa').value) / scale
            sigma = np.sqrt(msg.variance) / scale if msg.variance > 0 else None
        else:
            depth_z_up, sigma = extract_depth(msg)
        if not np.isfinite(depth_z_up):
            return
        if not self.depth_initialized:
            self.filter.p[2] = depth_z_up - (self.filter.R @ r_depth)[2]
            self.depth_initialized = True
        if self.method == 'eskf':
            self.filter.correct_depth(depth_z_up, r_depth, sigma=sigma)
        else:
            self.filter.update_depth(depth_z_up, r_depth)

    def _on_stereo(self, msg: PoseWithCovarianceStamped):
        if not self.is_initialized or self.method != 'eskf':
            return
        if self._sensor_time(msg) is None:
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
