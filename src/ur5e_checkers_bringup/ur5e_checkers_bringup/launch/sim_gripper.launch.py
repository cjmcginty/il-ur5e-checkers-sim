from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    TimerAction,
    SetEnvironmentVariable,
    IncludeLaunchDescription,
)
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    Command,
    EnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
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
            "ur5e_gripper.urdf.xacro"
        ]),
        " name:=ur5e",
        " ur_type:=ur5e",
        " tf_prefix:="
    ])

    # Make sure Gazebo can find custom plugins
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

    resource_path = PathJoinSubstitution([
        FindPackageShare("ur5e_checkers_bringup"),
        ".."
    ])

    robotiq_resource_path = "/workspaces/ur5e-checkers-irl/install/robotiq_description/share"

    set_resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=[
            resource_path,
            ":",
            robotiq_resource_path,
            ":",
            EnvironmentVariable("GZ_SIM_RESOURCE_PATH", default_value="")
        ]
    )

    # Launch Gazebo
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

    # Include MoveIt/Servo launch
    moveit_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("ur5e_checkers_bringup"),
                "launch",
                "ur_moveit_sim.launch.py"
            ])
        ),
        launch_arguments={
            "ur_type": "ur5e",
            "use_sim_time": "true",
            "launch_rviz": "false",
            "launch_servo": "true",
        }.items()
    )

    clock_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="clock_bridge",
        arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
        output="screen"
    )

    pose_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="pose_bridge",
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
                "model_states_topic": "/checkers/piece_states",
                "update_hz": 5.0,
                "use_sim_time": True,
            }
        ],
    )

    data_collection_node = Node(
        package="data_collection",
        executable="data_collection_node",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
            }
        ],
    )

    dqn_policy_node = Node(
        package="ur5e_checkers_bringup",
        executable="dqn_policy_node",
        output="screen",
        parameters=[
            {
                "board_state_topic": "/checkers/board_state",
                "legal_moves_topic": "/checkers/legal_moves",
                "selected_move_topic": "/checkers/selected_move",
                "model_path": "/workspaces/ur5e-checkers-irl/models/dqn_checkers.pt",
                "device": "auto",
                "publish_once_per_position": True,
                "republish_hz": 2.0,
            }
        ],
    )

    move_target_node = Node(
        package="ur5e_checkers_bringup",
        executable="move_target_node",
        name="move_target_node",
        output="screen",
        parameters=[
            {
                "selected_move_topic": "/checkers/selected_move",
                "piece_states_topic": "/checkers/piece_states",
                "move_target_topic": "/checkers/move_targets",
                "board_center_x": 0.6,
                "board_center_y": 0.0,
                "board_size": 0.40,
                "piece_z": 0.03,
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
            "forward_position_controller",
            "--controller-manager", "/controller_manager"
        ],
        output="screen"
    )

    spawn_gripper = ExecuteProcess(
        cmd=[
            "ros2", "run", "controller_manager", "spawner",
            "gripper_position_controller",
            "--controller-manager", "/controller_manager"
        ],
        output="screen"
    )

    set_start_pose = ExecuteProcess(
        cmd=[
            "ros2", "topic", "pub", "--once",
            "/forward_position_controller/commands",
            "std_msgs/msg/Float64MultiArray",
            "{data: [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]}"
        ],
        output="screen"
    )

    servo_command_type = ExecuteProcess(
        cmd=[
            "ros2", "service", "call",
            "/servo_node/switch_command_type",
            "moveit_msgs/srv/ServoCommandType",
            "{command_type: 1}"
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

    # Optional: launch teleop in a separate terminal window.
    # Uncomment this if you want the launch file to open teleop automatically.
    # This usually works better than launching teleop directly inside the same terminal.
    #
    # teleop_node = ExecuteProcess(
    #     cmd=[
    #         "gnome-terminal", "--", "bash", "-c",
    #         (
    #             "ros2 run teleop_twist_keyboard teleop_twist_keyboard "
    #             "--ros-args "
    #             "-p stamped:=true "
    #             "-p frame_id:=base_link "
    #             "-p use_sim_time:=true "
    #             "--remap cmd_vel:=/servo_node/delta_twist_cmds; "
    #             "exec bash"
    #         )
    #     ],
    #     output="screen"
    # )

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
        set_resource_path,

        gz,
        clock_bridge,
        pose_bridge,
        checkers_node,
        dqn_policy_node,
        move_target_node,
        rsp,
        static_tf,
        moveit_sim,

        TimerAction(period=2.0, actions=[spawn_robot]),
        TimerAction(period=6.0, actions=[spawn_jsb]),
        TimerAction(period=7.0, actions=[spawn_traj]),
        TimerAction(period=7.5, actions=[spawn_gripper]),
        TimerAction(period=8.5, actions=[set_start_pose]),

        # Let servo come up before switching command mode
        TimerAction(period=10.0, actions=[servo_command_type]),

        # Start data collection after the rest of the system is up
        TimerAction(period=10.5, actions=[data_collection_node]),

        TimerAction(period=11.0, actions=[board_spawn]),
        TimerAction(period=12.0, actions=red_spawns),
        TimerAction(period=13.0, actions=black_spawns),

        # Optional teleop startup
        # TimerAction(period=14.0, actions=[teleop_node]),
    ])