# ROS 2 Jazzy + desktop tools (rviz/rqt) + gazebo support
FROM osrf/ros:jazzy-desktop-full

SHELL ["/bin/bash", "-lc"]

# Basic dev deps
RUN apt-get update && apt-get install -y \
    python3-pip python3-venv python3-colcon-common-extensions \
    git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Basic dev deps
RUN apt-get update && apt-get install -y \
    python3-pip python3-venv python3-full \
    python3-colcon-common-extensions \
    git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a virtualenv for ML deps (avoids PEP 668)
ENV VENV_PATH=/opt/venv
RUN python3 -m venv ${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:${PATH}"

# Upgrade pip inside venv and install ML deps
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir numpy torch torchvision torchaudio


# Create workspace directory
ENV WS=/workspaces/ur5e-checkers-irl
WORKDIR ${WS}

# ROS defaults
RUN echo "source /opt/ros/jazzy/setup.bash" >> /etc/bash.bashrc
