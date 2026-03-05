# ─────────────────────────────────────────────────────────────────────────────
# Base: ROS2 Humble (includes rclpy + sensor/geometry/nav msgs out of the box)
# ─────────────────────────────────────────────────────────────────────────────
FROM ros:humble-ros-base

# Avoid interactive prompts during apt
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# ─────────────────────────────────────────────────────────────────────────────
# System dependencies
# ─────────────────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-opencv \
    # ROS2 message packages (sensor_msgs, geometry_msgs, nav_msgs)
    ros-humble-sensor-msgs \
    ros-humble-geometry-msgs \
    ros-humble-nav-msgs \
    # PostgreSQL client libs needed by psycopg
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────────────────────────────────────
# Python dependencies
# ─────────────────────────────────────────────────────────────────────────────
RUN pip3 install --no-cache-dir \
    eclipse-zenoh \
    psycopg[binary] \
    numpy

# ─────────────────────────────────────────────────────────────────────────────
# Copy application files
# (db_connect.py and coco_dict.py must be in the same build context)
# ─────────────────────────────────────────────────────────────────────────────
COPY object_detection_events.py .
COPY db_connect.py .
COPY coco_dict.py .

# ─────────────────────────────────────────────────────────────────────────────
# Source ROS2 setup and launch the pipeline
# ─────────────────────────────────────────────────────────────────────────────
CMD ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && python3 object_detection_events.py"]
