"""Launch full 3-D Sonar SLAM stack in ROS 2 Jazzy."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = get_package_share_directory('bruce_slam')
    default_config = os.path.join(pkg_share, 'config', 'bruce_slam.yaml')
    default_urdf = os.path.join(pkg_share, 'urdf', 'rov.urdf.xacro')
    default_rviz = os.path.join(pkg_share, 'rviz', 'bruce_slam3d.rviz')

    # Arguments
    use_sim_arg = DeclareLaunchArgument(
        'use_sim', default_value='false', description='Run synthetic underwater sensor simulator'
    )
    method_arg = DeclareLaunchArgument(
        'method', default_value='eskf', description='State estimation method: eskf or complementary'
    )
    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='false', description='Launch RViz2 visualization'
    )
    config_arg = DeclareLaunchArgument(
        'config', default_value=default_config, description='Path to parameters YAML file'
    )

    use_sim = LaunchConfiguration('use_sim')
    method = LaunchConfiguration('method')
    rviz = LaunchConfiguration('rviz')
    config_file = LaunchConfiguration('config')

    # Robot State Publisher (URDF)
    robot_description = ParameterValue(Command(['xacro ', default_urdf]), value_type=str)
    node_rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}]
    )

    # 1. State Estimation Node
    node_state_est = Node(
        package='bruce_slam',
        executable='state_estimation_node',
        output='screen',
        parameters=[config_file, {'method': method}]
    )

    # 2. Point Cloud Preprocessing Node
    node_preprocess = Node(
        package='bruce_slam',
        executable='pointcloud_preprocessing_node',
        output='screen',
        parameters=[config_file]
    )

    # 3. 3-D Sonar SLAM Node
    node_slam = Node(
        package='bruce_slam',
        executable='slam_node',
        output='screen',
        parameters=[config_file]
    )

    # 4. Simulation Publisher Node (Optional)
    node_sim = Node(
        package='bruce_slam',
        executable='sim_publisher_node',
        output='screen',
        condition=IfCondition(use_sim)
    )

    # 5. RViz2 (Optional)
    node_rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', default_rviz],
        condition=IfCondition(rviz)
    )

    return LaunchDescription([
        use_sim_arg,
        method_arg,
        rviz_arg,
        config_arg,
        node_rsp,
        node_state_est,
        node_preprocess,
        node_slam,
        node_sim,
        node_rviz
    ])
