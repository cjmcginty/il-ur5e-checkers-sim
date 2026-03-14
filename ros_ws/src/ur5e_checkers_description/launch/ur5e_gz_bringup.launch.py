from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = FindPackageShare("ur5e_checkers_description")

    xacro_file = PathJoinSubstitution([pkg_share, "urdf", "ur5e_gz.urdf.xacro"])
    controllers_file = PathJoinSubstitution([pkg_share, "config", "ur5e_controllers.yaml"])

    robot_description = ParameterValue(
        Command([
            "xacro ", xacro_file,
            " name:=ur5e",
            " tf_prefix:=",
            " parent:=world",
        ]),
        value_type=str
    )

    SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value="/opt/ros/jazzy/share:/workspaces/ur5e-checkers-irl/ros_ws/install"
    )

    # Launch Gazebo Harmonic (gz sim) using ros_gz_sim launch
    gz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("ros_gz_sim"),
                "launch",
                "gz_sim.launch.py"
            ])
        ),
        launch_arguments={
            "gz_args": "-r -s /opt/ros/jazzy/opt/gz_sim_vendor/share/gz/gz-sim8/worlds/empty.sdf"
        }.items()
    )

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description}],
    )

    # Spawn the robot into Gazebo from robot_description topic
    spawn = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-name", "ur5e",
            "-topic", "/robot_description",
            "-x", "0", "-y", "0", "-z", "0.2",
            "--ros-args", "-r", "__node:=create_ur5e",
        ],
    )


    # Load controllers
    jsb = Node(
        package="controller_manager",
        executable="spawner",
        output="screen",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
    )

    arm = Node(
        package="controller_manager",
        executable="spawner",
        output="screen",
        arguments=["arm_controller", "--controller-manager", "/controller_manager"],
    )

    desc_pub = Node(
        package="ur5e_checkers_description",
        executable="robot_description_publisher.py",
        output="screen",
        parameters=[{"robot_description": robot_description}],
    )


    return LaunchDescription([
        gz_launch,
        rsp,
        desc_pub,
        spawn,
        jsb,
        arm,
    ])
