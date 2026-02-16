# ROS 2 Jazzy + desktop tools (rviz/rqt) + gazebo support
FROM osrf/ros:jazzy-desktop-full

SHELL ["/bin/bash", "-lc"]

# Basic dev deps
RUN apt-get update && apt-get install -y \
    python3-pip python3-venv python3-colcon-common-extensions \
    git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python ML deps (start minimal; expand later)
RUN pip3 install --no-cache-dir \
    numpy \
    torch \
    torchvision \
    torchaudio

# Create workspace directory
ENV WS=/workspaces/ur5e-checkers-irl
WORKDIR ${WS}

# ROS defaults
RUN echo "source /opt/ros/jazzy/setup.bash" >> /etc/bash.bashrc
