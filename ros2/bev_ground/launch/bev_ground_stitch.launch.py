# Metric bird's-eye ground mosaic on the hardware-triggered 4x IMX296 rig.
#
# Consumes whole synchronised camera sets from argus_capture_node; it does not open the
# cameras itself. Start the capture node first (scripts/vo/... or the compose service),
# then this.
#
# It will refuse to start until config/rig/ground_plane.yaml carries a MEASURED plane.
# That is deliberate: the rig height and tilt are the terms every ground projection
# depends on, and an error in them is indistinguishable in the output from a camera
# calibration error. For a bench look before that measurement exists:
#
#   ros2 launch bev_ground bev_ground_stitch.launch.py \
#       allow_unmeasured_plane:=true provisional_height_m:=0.28
#
# ...and read nothing metric off the result.
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    args = [
        DeclareLaunchArgument('params_file', default_value=PathJoinSubstitution(
            [FindPackageShare('bev_ground'), 'config', 'ground_stitch_params.yaml'])),
        DeclareLaunchArgument('allow_unmeasured_plane', default_value='false'),
        DeclareLaunchArgument('provisional_height_m', default_value='0.0'),
        DeclareLaunchArgument('resolution_m_per_px', default_value='0.01'),
    ]
    return LaunchDescription(args + [
        Node(
            package='bev_ground',
            executable='bev_ground_stitch_node',
            name='bev_ground_stitch',
            output='screen',
            parameters=[
                LaunchConfiguration('params_file'),
                {
                    'allow_unmeasured_plane': LaunchConfiguration('allow_unmeasured_plane'),
                    'provisional_height_m': LaunchConfiguration('provisional_height_m'),
                    'resolution_m_per_px': LaunchConfiguration('resolution_m_per_px'),
                },
            ],
        ),
    ])
