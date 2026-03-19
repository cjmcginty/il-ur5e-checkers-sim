from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command, EnvironmentVariable
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    spawn_board = LaunchConfiguration("spawn_board")

    world_path = PathJoinSubstitution([
        FindPackageShare("ur5e_checkers_bringup"),
        "worlds",
        "checkers_world.sdf"
    ])

    robot_description = Command([
        "xacro ",
        PathJoinSubstitution([
            FindPackageShare("ur5e_checkers_bringup"),
            "urdf",
            "ur5e_gz.urdf.xacro"
        ]),
        " name:=ur5e",
        " ur_type:=ur5e",
        " tf_prefix:="
    ])

    # make sure Gazebo can find custom plugins
    plugin_path = PathJoinSubstitution([
        FindPackageShare("checkers_gz_plugins"),
        "..",
        "..",
        "lib"
    ])

    set_plugin_path = SetEnvironmentVariable(
        name="GZ_SIM_SYSTEM_PLUGIN_PATH",
        value=[
            plugin_path,
            ":",
            EnvironmentVariable("GZ_SIM_SYSTEM_PLUGIN_PATH", default_value="")
        ]
    )

    # Launch Gazebo directly since this path worked in manual testing
    gz = ExecuteProcess(
        cmd=[
            "gz", "sim",
            "-r",
            "-v", "4",
            world_path
        ],
        additional_env={
            "GZ_SIM_SYSTEM_PLUGIN_PATH": [
                plugin_path,
                ":",
                EnvironmentVariable("GZ_SIM_SYSTEM_PLUGIN_PATH", default_value="")
            ]
        },
        output="screen"
    )

    clock_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
        output="screen"
    )

    pose_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/world/checkers_world/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V"
        ],
        output="screen",
    )

    checkers_node = Node(
        package="ur5e_checkers_bringup",
        executable="checkers_game_node",
        output="screen",
        parameters=[
            {
                "model_states_topic": "/world/checkers_world/dynamic_pose/info",
                "update_hz": 5.0,
            }
        ],
    )

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[
            {
                "robot_description": robot_description,
                "use_sim_time": True
            }
        ],
        output="screen"
    )

    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=["0", "0", "0", "0", "0", "0", "world", "base_link"],
        output="screen"
    )

    spawn_robot = ExecuteProcess(
        cmd=[
            "ros2", "run", "ros_gz_sim", "create",
            "-name", "ur5e",
            "-topic", "robot_description",
            "-x", "0", "-y", "0", "-z", "0.0"
        ],
        output="screen"
    )

    spawn_jsb = ExecuteProcess(
        cmd=[
            "ros2", "run", "controller_manager", "spawner",
            "joint_state_broadcaster",
            "--controller-manager", "/controller_manager"
        ],
        output="screen"
    )

    spawn_traj = ExecuteProcess(
        cmd=[
            "ros2", "run", "controller_manager", "spawner",
            "ur5e_arm_controller",
            "--controller-manager", "/controller_manager"
        ],
        output="screen"
    )

    board_spawn = ExecuteProcess(
        cmd=[
            "ros2", "run", "ros_gz_sim", "create",
            "-name", "checkers_board",
            "-file", PathJoinSubstitution([
                FindPackageShare("ur5e_checkers_bringup"),
                "models", "checkers_board", "model.sdf"
            ]),
            "-x", "0.6", "-y", "0.0", "-z", "0.01"
        ],
        output="screen"
    )

    # Board geometry assumptions
    board_center_x = 0.6
    board_center_y = 0.0
    board_size = 0.40
    square = board_size / 8.0
    piece_z = 0.03

    red_spawns = []
    black_spawns = []

    # Red pieces: top 3 rows
    red_count = 1
    for row in range(3):
        y = board_center_y + (board_size / 2.0) - (row + 0.5) * square
        for col in range(8):
            if (row + col) % 2 == 1:
                x = board_center_x - (board_size / 2.0) + (col + 0.5) * square
                red_spawns.append(
                    ExecuteProcess(
                        cmd=[
                            "ros2", "run", "ros_gz_sim", "create",
                            "-name", f"red_checker_{red_count}",
                            "-file", PathJoinSubstitution([
                                FindPackageShare("ur5e_checkers_bringup"),
                                "models", "red_checker", "model.sdf"
                            ]),
                            "-x", str(x), "-y", str(y), "-z", str(piece_z)
                        ],
                        output="screen"
                    )
                )
                red_count += 1

    # Black pieces: bottom 3 rows
    black_count = 1
    for row in range(5, 8):
        y = board_center_y + (board_size / 2.0) - (row + 0.5) * square
        for col in range(8):
            if (row + col) % 2 == 1:
                x = board_center_x - (board_size / 2.0) + (col + 0.5) * square
                black_spawns.append(
                    ExecuteProcess(
                        cmd=[
                            "ros2", "run", "ros_gz_sim", "create",
                            "-name", f"black_checker_{black_count}",
                            "-file", PathJoinSubstitution([
                                FindPackageShare("ur5e_checkers_bringup"),
                                "models", "black_checker", "model.sdf"
                            ]),
                            "-x", str(x), "-y", str(y), "-z", str(piece_z)
                        ],
                        output="screen"
                    )
                )
                black_count += 1

    return LaunchDescription([
        DeclareLaunchArgument(
            "spawn_board",
            default_value="true",
            description="Spawn a simple static checkers board model"
        ),

        set_plugin_path,
        gz,
        clock_bridge,
        pose_bridge,
        checkers_node,
        rsp,
        static_tf,

        TimerAction(period=2.0, actions=[spawn_robot]),
        TimerAction(period=3.0, actions=[board_spawn]),
        TimerAction(period=4.0, actions=red_spawns),
        TimerAction(period=5.0, actions=black_spawns),
        TimerAction(period=6.5, actions=[spawn_jsb]),
        TimerAction(period=7.5, actions=[spawn_traj]),
    ])