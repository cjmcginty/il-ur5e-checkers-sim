from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
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
            "ur5e_gz.urdf.xacro"
        ]),
        " name:=ur5e",
        " ur_type:=ur5e",
        " tf_prefix:="
    ])

    # ✅ Correct: pass the world path as part of gz_args (single launch arg)
    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("ros_gz_sim"),
                "launch",
                "gz_sim.launch.py"
            ])
        ),
        launch_arguments={
            "gz_args": ["-r -v4 ", world_path]
        }.items()
    )

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_description}],
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

    return LaunchDescription([
        DeclareLaunchArgument(
            "spawn_board",
            default_value="true",
            description="Spawn a simple static checkers board model"
        ),

        gz,
        rsp,
        static_tf,

        TimerAction(period=2.0, actions=[spawn_robot]),
        TimerAction(period=3.0, actions=[board_spawn]),
        TimerAction(period=6.0, actions=[spawn_jsb]),
        TimerAction(period=7.0, actions=[spawn_traj]),
    ])
