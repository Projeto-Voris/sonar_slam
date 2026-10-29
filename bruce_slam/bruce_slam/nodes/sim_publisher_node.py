"""Simulation publisher node: streams synthetic sensor data for live ROS 2 testing."""
import rclpy
from rclpy.node import Node
import numpy as np

from sensor_msgs.msg import Imu, PointCloud2
from geometry_msgs.msg import TwistWithCovarianceStamped, PoseWithCovarianceStamped
from std_msgs.msg import Header

from ..geometry import euler_to_rot, rot_to_quat
from ..sim.rov_sim import ROVSimulator, SimConfig
from ..utils.conversions import numpy_to_pointcloud2
from ..utils.topics import sensor_qos


class SimPublisherNode(Node):
    """Generates and streams synthetic underwater sensor topics at realistic rates."""

    def __init__(self):
        super().__init__('sim_publisher_node')

        self.declare_parameter('simulation_speed', 1.0)
        self.sim = ROVSimulator(SimConfig())
        self.start_time = self.get_clock().now()
        self.last_sim_t = 0.0

        # Publishers
        self.pub_imu = self.create_publisher(Imu, '/imu/data_raw', sensor_qos)
        self.pub_dvl = self.create_publisher(TwistWithCovarianceStamped, '/dvl/velocity', sensor_qos)
        self.pub_depth = self.create_publisher(PoseWithCovarianceStamped, '/altitude/depth', sensor_qos)
        self.pub_stereo = self.create_publisher(PoseWithCovarianceStamped, '/stereo/pose', sensor_qos)
        self.pub_sonar = self.create_publisher(PointCloud2, '/sonar_3d/points', sensor_qos)

        # Timers matching sensor rates
        self.create_timer(1.0 / self.sim.cfg.imu_rate, self._tick_imu)
        self.create_timer(1.0 / self.sim.cfg.depth_rate, self._tick_depth)
        self.create_timer(1.0 / self.sim.cfg.dvl_rate, self._tick_dvl)
        self.create_timer(1.0 / self.sim.cfg.stereo_rate, self._tick_stereo)
        self.create_timer(1.0 / self.sim.cfg.sonar_rate, self._tick_sonar)

        self.get_logger().info('sim_publisher_node started. Streaming synthetic sensor data.')

    def _get_sim_time(self) -> float:
        now = self.get_clock().now()
        dur = (now - self.start_time).nanoseconds * 1e-9
        return dur * self.get_parameter('simulation_speed').value

    def _tick_imu(self):
        t = self._get_sim_time()
        gyro, accel = self.sim.sample_imu(t)

        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'imu_link'
        msg.angular_velocity.x = float(gyro[0])
        msg.angular_velocity.y = float(gyro[1])
        msg.angular_velocity.z = float(gyro[2])
        msg.linear_acceleration.x = float(accel[0])
        msg.linear_acceleration.y = float(accel[1])
        msg.linear_acceleration.z = float(accel[2])
        self.pub_imu.publish(msg)

    def _tick_dvl(self):
        t = self._get_sim_time()
        v = self.sim.sample_dvl(t)

        msg = TwistWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'dvl_link'
        msg.twist.twist.linear.x = float(v[0])
        msg.twist.twist.linear.y = float(v[1])
        msg.twist.twist.linear.z = float(v[2])
        cov = np.eye(6) * (self.sim.cfg.dvl_noise**2)
        msg.twist.covariance = cov.flatten().tolist()
        self.pub_dvl.publish(msg)

    def _tick_depth(self):
        t = self._get_sim_time()
        depth = self.sim.sample_depth(t)

        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'depth_link'
        msg.pose.pose.position.z = float(depth)
        cov = np.eye(6) * (self.sim.cfg.depth_noise**2)
        msg.pose.covariance = cov.flatten().tolist()
        self.pub_depth.publish(msg)

    def _tick_stereo(self):
        t = self._get_sim_time()
        dt = t - self.last_sim_t
        self.last_sim_t = t
        p, yaw = self.sim.sample_stereo(t, max(dt, 0.01))

        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'stereo_link'
        msg.pose.pose.position.x = float(p[0])
        msg.pose.pose.position.y = float(p[1])
        msg.pose.pose.position.z = float(p[2])
        R = euler_to_rot(0.0, 0.0, yaw)
        q = rot_to_quat(R)
        msg.pose.pose.orientation.w = float(q[0])
        msg.pose.pose.orientation.x = float(q[1])
        msg.pose.pose.orientation.y = float(q[2])
        msg.pose.pose.orientation.z = float(q[3])
        cov = np.eye(6) * (self.sim.cfg.stereo_noise_pos**2)
        cov[5, 5] = 0.02**2
        msg.pose.covariance = cov.flatten().tolist()
        self.pub_stereo.publish(msg)

    def _tick_sonar(self):
        t = self._get_sim_time()
        pts = self.sim.sample_sonar(t)
        cloud_msg = numpy_to_pointcloud2(pts, self.get_clock().now().to_msg(), frame_id='sonar_link')
        self.pub_sonar.publish(cloud_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SimPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
