FROM osrf/ros:jazzy-desktop-full

SHELL ["/bin/bash", "-lc"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get dist-upgrade -y && \
    apt-get install -y --no-install-recommends \
      build-essential \
      git \
      curl \
      python3-full \
      python3-venv \
      python3-pip \
      python3-colcon-common-extensions \
      python3-rosdep \
      nano \
      less \
      sudo \
      ros-jazzy-ur-description \
      ros-jazzy-ur-moveit-config \
      ros-jazzy-ur-robot-driver \
      ros-jazzy-ros-gz-sim \
      ros-jazzy-ros-gz-bridge \
      ros-jazzy-xacro \
      ros-jazzy-ros2-control \
      ros-jazzy-ros2-controllers \
      ros-jazzy-gz-ros2-control \
      ros-jazzy-teleop-twist-keyboard \
      ros-jazzy-moveit-servo \
      ros-jazzy-moveit-configs-utils && \
    rm -rf /var/lib/apt/lists/*

RUN echo "ubuntu ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    echo "root ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN pip3 install --break-system-packages --no-cache-dir \
      catkin_pkg \
      empy \
      lark-parser \
      pytest \
      numpy \
      torch \
      torchvision \
      torchaudio

WORKDIR /workspaces/ur5e-checkers-irl

RUN echo "source /opt/ros/jazzy/setup.bash" >> /etc/bash.bashrc