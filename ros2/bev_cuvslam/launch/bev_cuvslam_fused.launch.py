"""Launch the fused zero-copy Argus -> cuVSLAM VO node with params from config/fused_vo_params.yaml.

  ros2 launch bev_cuvslam bev_cuvslam_fused.launch.py

Override a param without editing the yaml, e.g.:
  ros2 launch bev_cuvslam bev_cuvslam_fused.launch.py params:=/abs/other_params.yaml
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('bev_cuvslam'), 'config', 'fused_vo_params.yaml')
    params = LaunchConfiguration('params')
    return LaunchDescription([
        DeclareLaunchArgument('params', default_value=default_params,
                              description='Path to the fused-VO params yaml'),
        Node(
            package='bev_cuvslam',
            executable='bev_cuvslam_fused_node',
            name='bev_cuvslam_fused',
            output='screen',
            parameters=[params],
        ),
    ])
