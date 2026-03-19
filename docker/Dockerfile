# ROS 2 Jazzy + Desktop tools
FROM osrf/ros:jazzy-desktop-full

SHELL ["/bin/bash", "-lc"]

# System + ROS + Gazebo( Harmonic via vendor pkgs ) + ros2_control + bridge
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    python3-full python3-venv python3-pip \
    python3-colcon-common-extensions python3-rosdep \
    nano less \
    ros-jazzy-ur-description \
    ros-jazzy-ur-moveit-config \
    ros-jazzy-ur-robot-driver \
    #ros-jazzy-ros-gz \
    ros-jazzy-ros-gz-sim \
    ros-jazzy-ros-gz-bridge \
    ros-jazzy-xacro \
    ros-jazzy-ros2-control \
    ros-jazzy-ros2-controllers \
    ros-jazzy-gz-ros2-control \
    ros-jazzy-teleop-twist-keyboard \
    #ros-jazzy-moveit \
    ros-jazzy-moveit-servo \
    #ros-jazzy-moveit-setup-assistant \
    ros-jazzy-moveit-configs-utils \
 && rm -rf /var/lib/apt/lists/*

# Python deps (ROS build helpers + ML) - install into system python so ros2 nodes can import them
RUN pip3 install --break-system-packages --no-cache-dir \
      catkin_pkg empy lark-parser pytest \
      numpy torch torchvision torchaudio

# Workspace
WORKDIR /workspaces/ur5e-checkers-irl

# Auto-source ROS
RUN echo "source /opt/ros/jazzy/setup.bash" >> /etc/bash.bashrc