"""Launch the surround panorama node with params from config/panorama_params.yaml.

  ros2 launch bev_cuvslam bev_panorama.launch.py
  ros2 launch bev_cuvslam bev_panorama.launch.py params:=/abs/other_params.yaml
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('bev_cuvslam'), 'config', 'panorama_params.yaml')
    params = LaunchConfiguration('params')
    return LaunchDescription([
        DeclareLaunchArgument('params', default_value=default_params,
                              description='Path to the panorama params yaml'),
        Node(
            package='bev_cuvslam',
            executable='bev_panorama_node',
            name='bev_panorama',
            output='screen',
            parameters=[params],
        ),
    ])
