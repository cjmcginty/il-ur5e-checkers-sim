# ROS 2 Jazzy + desktop tools (rviz/rqt) + gazebo support
FROM osrf/ros:jazzy-desktop-full

SHELL ["/bin/bash", "-lc"]

# System deps
RUN apt-get update && apt-get install -y \
    python3-full python3-venv \
    git build-essential \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# Create a Python venv for ML deps (avoids PEP 668)
ENV VENV_PATH=/opt/venv
RUN python3 -m venv ${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:${PATH}"

# ML deps inside venv
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir numpy torch torchvision torchaudio

# Workspace
WORKDIR /workspaces/ur5e-checkers-irl

# Source ROS automatically
RUN echo "source /opt/ros/jazzy/setup.bash" >> /etc/bash.bashrc
