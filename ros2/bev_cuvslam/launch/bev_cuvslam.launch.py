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
        # 300 is the header's real-time figure and it CAPS the graph: on a 57 s replay it was
        # reached at t=41.6 s, after which GetAllSlamPoses simply stops growing and the
        # optimised trajectory silently ends mid-run. 0 = unlimited, which is what an offline
        # resim wants. See add-replay-visual-diagnostics 1.7h.
        DeclareLaunchArgument('slam_max_map_size', default_value='300'),
        DeclareLaunchArgument('slam_throttling_ms', default_value='0'),
        # The PROMOTED SLAM map, as a cloud. Separate from publish_landmarks, which is the
        # odometry track dump and says nothing about what loop closure can match against.
        # See add-replay-visual-diagnostics 1.7l (cuVSLAM issue #136).
        DeclareLaunchArgument('publish_map_landmarks', default_value='false'),
        # One row per set: Track() us, callback us, promoted-map size. Empty = off. This is
        # the only way to see a cost trend - the periodic log prints a windowed maximum, and
        # a replay's message interval is floored by the replay rate (issue #77).
        DeclareLaunchArgument('timing_csv', default_value=''),
        # cuVSLAM's OWN log, off by default (SetVerbosity(0) is the library default).
        # 2 = Warning, which is the level that surfaces delay_warning_queue_size: SLAM runs
        # its backend on a background thread fed a keyframe queue, and when that queue grows
        # past 10 the poses and loop closures it reports refer to an increasingly OLD point
        # on the trajectory. That is the backend falling behind the frontend - the #77
        # symptom - and we have never had it switched on. 1=Error 2=Warning 3=Message.
        # THE ONLY WAY TO TIME THE SLAM BACKEND. With this false (the library default),
        # Slam::Track converts observations, pushes a keyframe onto a queue and returns —
        # loop-closure matching and pose-graph optimisation then run on cuVSLAM's own worker
        # thread, where no timer of ours can see them. Anything measured about "SLAM cost"
        # with this false is the ENQUEUE, not the optimisation. True runs it inline, which is
        # far slower per set and needs a correspondingly slow replay, but it is the only
        # configuration in which the backend's growth with map size is observable.
        DeclareLaunchArgument('slam_sync_mode', default_value='false'),
        # Bound the MAP (not the pose graph). 100 m is the library default and is absurd
        # indoors; far points are also the worst-triangulated ones on a 0.1 m virtual
        # baseline. 0 = keep the library default. See 1.7o.
        DeclareLaunchArgument('slam_max_landmarks_distance', default_value='0.0'),
        DeclareLaunchArgument('slam_map_cell_size', default_value='0.0'),
        DeclareLaunchArgument('cuvslam_verbosity', default_value='0'),
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
                'slam_max_map_size': ParameterValue(
                    LaunchConfiguration('slam_max_map_size'), value_type=int),
                'slam_throttling_ms': ParameterValue(
                    LaunchConfiguration('slam_throttling_ms'), value_type=int),
                'image_qos': LaunchConfiguration('image_qos'),
                'image_qos_depth': ParameterValue(LaunchConfiguration('image_qos_depth'),
                                                  value_type=int),
                'match_history': 8,
                'publish_landmarks': LaunchConfiguration('publish_landmarks'),
                'publish_observations': LaunchConfiguration('publish_observations'),
                'publish_map_landmarks': LaunchConfiguration('publish_map_landmarks'),
                'timing_csv': LaunchConfiguration('timing_csv'),
                'cuvslam_verbosity': ParameterValue(
                    LaunchConfiguration('cuvslam_verbosity'), value_type=int),
                'slam_sync_mode': ParameterValue(
                    LaunchConfiguration('slam_sync_mode'), value_type=bool),
                'slam_max_landmarks_distance': ParameterValue(
                    LaunchConfiguration('slam_max_landmarks_distance'), value_type=float),
                'slam_map_cell_size': ParameterValue(
                    LaunchConfiguration('slam_map_cell_size'), value_type=float),
            }],
        ),
    ])
