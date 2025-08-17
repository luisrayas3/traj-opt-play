FROM osrf/ros:noetic-desktop-full

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    libeigen3-dev \
    libboost-all-dev \
    libbullet-dev \
    libfcl-dev \
    libassimp-dev \
    liboctomap-dev \
    python3-dev \
    python3-numpy \
    python3-scipy \
    python3-matplotlib \
    python3-yaml \
    ros-noetic-xacro \
    ros-noetic-ur-description \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
WORKDIR /app

# Update pip and install Python packages
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install "numpy>=1.19,<1.24" && \
    python3 -m pip install tesseract-robotics && \
    python3 -m pip install \
        scipy \
        matplotlib \
        pybullet \
        jupyter \
        ipython

# Copy source files
COPY src/ ./src/

# Set up ROS environment to source entrypoint
RUN echo "source /app/src/entrypoint.sh" >> ~/.bashrc

ENTRYPOINT ["./src/entrypoint.sh"]
CMD ["python3", "src/main.py"]