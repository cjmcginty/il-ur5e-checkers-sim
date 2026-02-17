# ROS 2 Jazzy
FROM osrf/ros:jazzy-desktop-full

SHELL ["/bin/bash", "-lc"]

# System + ROS + GZ deps (Gazebo comes via ros-gz vendor packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git curl \
    python3-full python3-venv python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    ros-jazzy-ros-gz \
    ros-jazzy-ros-gz-sim \
    ros-jazzy-ros-gz-bridge \
    ros-jazzy-ros-gz-control \
    ros-jazzy-ros2-control \
    ros-jazzy-ros2-controllers \
    ros-jazzy-xacro \
 && rm -rf /var/lib/apt/lists/*

# Create venv
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Python deps needed for colcon/ament packages
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir catkin_pkg empy lark-parser pytest

# ML deps (start minimal; recommend CPU-only for stability)
RUN pip install --no-cache-dir numpy

# Workspace
WORKDIR /workspaces/ur5e-checkers-irl

# Auto-source ROS
RUN echo "source /opt/ros/jazzy/setup.bash" >> /etc/bash.bashrc
