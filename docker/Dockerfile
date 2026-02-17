# ROS 2 Jazzy
FROM osrf/ros:jazzy-desktop-full

SHELL ["/bin/bash", "-lc"]

# System + ROS + GZ deps
RUN apt-get update && apt-get install -y \
    python3-full python3-venv \
    python3-colcon-common-extensions \
    git build-essential \
    gz-harmonic \
    ros-jazzy-ros-gz \
    ros-jazzy-ros-gz-sim \
    ros-jazzy-ros-gz-bridge \
    ros-jazzy-xacro \
    ros-jazzy-ur-description \
    && rm -rf /var/lib/apt/lists/*

# Create venv
RUN apt-get update && apt-get install -y python3-full python3-venv && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Install Python deps needed for colcon/ament packages + ML
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
      catkin_pkg empy lark-parser pytest \
      numpy torch torchvision torchaudio

# Make venv writable by your default UID (1000)
RUN chown -R 1000:1000 /opt/venv

# Workspace
WORKDIR /workspaces/ur5e-checkers-irl

# Auto-source ROS
RUN echo "source /opt/ros/jazzy/setup.bash" >> /etc/bash.bashrc
