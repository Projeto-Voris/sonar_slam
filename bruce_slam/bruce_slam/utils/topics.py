"""QoS profiles and default topic names."""
from rclpy.qos import (
    QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
)

# Best-effort QoS for high-rate sensor streams (IMU, DVL, Sonar)
sensor_qos = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=20,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE
)

# Reliable QoS for internal estimates and triggers
reliable_qos = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE
)

# Latched QoS (Transient Local) for map, paths, and persistent visualization
latched_qos = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
)
