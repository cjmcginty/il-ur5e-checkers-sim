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
    && rm -rf /var/lib/apt/lists/*

# Create Python venv (avoids PEP 668)
ENV VENV_PATH=/opt/venv
RUN python3 -m venv ${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:${PATH}"

# ML deps
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir numpy torch torchvision torchaudio

# Workspace
WORKDIR /workspaces/ur5e-checkers-irl

# Auto-source ROS
RUN echo "source /opt/ros/jazzy/setup.bash" >> /etc/bash.bashrc
