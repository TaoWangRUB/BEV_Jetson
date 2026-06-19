from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='bev_cuvslam',
            executable='cuvslam_multicam_node',
            name='cuvslam_multicam',
            output='screen',
            parameters=[{
                # absolute paths recommended at runtime (these are repo-relative defaults)
                'calib_dir': 'config/calib/1640x1232',
                'rig_extrinsics': 'config/rig/rig_extrinsics.yaml',
                'cameras': ['cam1', 'cam2', 'cam3', 'cam4'],
                'image_topics': [
                    '/cam1/image_raw', '/cam2/image_raw',
                    '/cam3/image_raw', '/cam4/image_raw',
                ],
                'odom_frame': 'odom',
                'base_frame': 'base_link',
                'sync_slop_ms': 20,   # IMX219 has no hw trigger -> generous slop
            }],
        ),
    ])
