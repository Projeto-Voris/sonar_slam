from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'bruce_slam'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.xacro')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Daniel Regner',
    maintainer_email='daniel_regner@hotmail.com',
    description='3-D Sonar SLAM for ROVs in ROS 2 Jazzy',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'state_estimation_node = bruce_slam.nodes.state_estimation_node:main',
            'pointcloud_preprocessing_node = bruce_slam.nodes.pointcloud_preprocessing_node:main',
            'slam_node = bruce_slam.nodes.slam_node:main',
            'sim_publisher_node = bruce_slam.nodes.sim_publisher_node:main',
            'compare_icp = bruce_slam.nodes.compare_icp_node:main',
        ],
    },
)
