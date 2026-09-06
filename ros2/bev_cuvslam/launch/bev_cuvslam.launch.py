# cuVSLAM multicam VO on the hardware-triggered 4x IMX296 rig.
#
# The node subscribes to the four fisheyes and carves each into two virtual pinholes
# before cuVSLAM sees anything - the raw cameras are ~192 deg and cuVSLAM's equidistant
# model is capped below 180, so this is required rather than preferred. See docs/cuvslam_tx2.md.
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        # Off by default: the landmark export slows an already Track()-bound node, so keep
        # it for visualisation runs and leave the §5 rate measurement on the plain path.
        DeclareLaunchArgument('publish_landmarks', default_value='false'),
        DeclareLaunchArgument('publish_observations', default_value='false'),
        # 'sensor_data' (BEST_EFFORT) is the live-rig default: a slow tracker must never
        # back-pressure the camera. Bag replay wants 'reliable' - see the node comment and
        # retarget-vo-to-imx296-rig 5.10.
        # Loop closure. Off by default: it forces the observation/landmark export
        # (Slam::Track consumes Odometry::State, and GetState throws without it) and adds a
        # pose graph, which the TX2 has no headroom for. See add-replay-visual-diagnostics 1.7.
        DeclareLaunchArgument('enable_slam', default_value='false'),
        DeclareLaunchArgument('image_qos', default_value='sensor_data'),
        DeclareLaunchArgument('image_qos_depth', default_value='10'),
        Node(
            package='bev_cuvslam',
            executable='cuvslam_multicam_node',
            name='cuvslam_multicam',
            output='screen',
            parameters=[{
                # Absolute paths recommended at runtime; these are repo-relative defaults.
                'calib_dir': 'config/calib/imx296_1456x1088',
                # Ring-closed extrinsics. The node reads the rig_in_cam1 block, not the
                # pairwise ones - a multi-camera solver wants one rigid rig.
                'rig_extrinsics': 'config/rig/rig_extrinsics_imx296.yaml',
                'virtual_stereo': 'config/rig/virtual_stereo_imx296.yaml',
                'cameras': ['cam1', 'cam2', 'cam3', 'cam4'],
                'image_topics': [
                    '/cam1/image_raw', '/cam2/image_raw',
                    '/cam3/image_raw', '/cam4/image_raw',
                ],
                'odom_frame': 'odom',
                'base_frame': 'cam1_optical_frame',   # NOT base_link: cam1 optical, 180-rolled. See 3R.16b.
                # A set whose frames span more than this is not a set. cuVSLAM's own
                # Multicamera gate is 1 ms and the triggered rig measures 1 us, so
                # anything near this limit is a trigger fault - do NOT widen it to make
                # sets appear. The bundler that used to do exactly that is gone.
                'max_skew_us': 1000,
                'enable_slam': ParameterValue(LaunchConfiguration('enable_slam'),
                                              value_type=bool),
                'image_qos': LaunchConfiguration('image_qos'),
                'image_qos_depth': ParameterValue(LaunchConfiguration('image_qos_depth'),
                                                  value_type=int),
                'match_history': 8,
                'publish_landmarks': LaunchConfiguration('publish_landmarks'),
                'publish_observations': LaunchConfiguration('publish_observations'),
            }],
        ),
    ])
