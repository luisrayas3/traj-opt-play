#!/bin/bash

# Source ROS setup
source /opt/ros/noetic/setup.bash

# Set environment variables
export PYTHONPATH=/app/src:$PYTHONPATH

exec "$@"