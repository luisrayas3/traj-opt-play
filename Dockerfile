FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# Create workspace
WORKDIR /app

# Install git and OpenGL libraries for viewer
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Clone ur_description for mesh files into /app
RUN git clone --depth 1 --branch noetic-devel \
    https://github.com/ros-industrial/universal_robot.git \
    /app/universal_robot

# Install Python packages
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install \
        "numpy>=1.19,<1.24" \
        tesseract-robotics==0.5.1 \
        tesseract-robotics-viewer==0.5.0 \
        xacro

# Copy source files
COPY src/ ./src/

CMD ["python3", "src/main.py"]
