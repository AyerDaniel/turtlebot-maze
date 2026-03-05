#!/usr/bin/env python3
"""
Object-detection pipeline from TurtleBot3 simulation through Zenoh into PostgreSQL
using an image queue to pair images with detections.
"""

import zenoh
import json
import time
import psycopg
import cv2
import numpy as np
from queue import Queue
import uuid
import math
import rclpy

from datetime import datetime, timezone

# Import packages for deserializing Zenoh output from ROS2 sensor_msgs/msg/Image.
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose

from datetime import datetime

from db_connect import *

class semantic_pipeline:
    def __init__(self, table, queue_size=10, show_camera=True):
        self.table = table
        self.show_camera = show_camera
        self.image_queue = Queue(maxsize=queue_size)

        # Initialize the data structure
        self.data = {
            "schema": "maze.detection.v1",
            "event_id": str(uuid.uuid4()),  # unique event ID
            "run_id": str(uuid.uuid4()),    # unique run ID
            "robot_id": "tb3_sim",
            "sequence": int(),
            "image": {},
            "odometry": {},
            "tf": {},
            "detections": {}
        }
        # Zenoh connection
        conf = zenoh.Config()
        conf.insert_json5("connect/endpoints", '["tcp/localhost:7447"]')

        try:
            self.session = zenoh.open(conf)

        except Exception as e:

            print(f"Failure creating Zenoh session:\nError: {e}", flush=True)
            raise

        print("Connected to Zenoh router.", flush=True)

        # Subscribe to demo-SLAM.  (Container demo-slam must be running and publishing to Zenoh).
        self.sub_slam = self.session.declare_subscriber("tb/slam/pose", self.slam_pose_callback)
        print(f"Subscribed to: tb/slam/pose")

        # Subscribe to demo-SLAM.  (Container demo-slam must be running and publishing to Zenoh).
        self.sub_slam = self.session.declare_subscriber("tb/slam/status", self.slam_status_callback)
        print(f"Subscribed to: tb/slam/status")

        # Subscribe to odom.
        self.sub_robot_state = self.session.declare_subscriber("odom", self.odom_callback)
        print(f"Subscribed to: tb/odom")

        # Subscribe to TF
        self.sub_tf = self.session.declare_subscriber("tf_static", self.tf_callback)
        print(f"Subscribed to: tf")

    def slam_pose_callback(self, sample):
        
        # Extract the raw byte data
        msg = sample.payload.to_bytes().decode()

        print(msg)

    def slam_status_callback(self, sample):

        # Extract the raw byte data
        msg = sample.payload.to_bytes().decode()

        print(msg)

    def tf_callback(self, sample):

        try:

            # Extract the raw byte data
            raw_data = sample.payload.to_bytes()
 
            # Deserialize the message
            msg = deserialize_message(raw_data, TransformStamped)

            # Get the base_frame, camera_frame, and the transformation matrix
            self.data["tf"]["base_frame"] = msg.header.frame_id
            self.data["tf"]["camera_frame"] = msg.child_frame_id
            self.data["tf"]["t_base_camera"] = [msg.transform.translation.x,
                                                msg.transform.translation.y,
                                                msg.transform.translation.z,
                                                msg.transform.rotation.x,
                                                msg.transform.rotation.y,
                                                msg.transform.rotation.z,
                                                msg.transform.rotation.w]
            
            # Set this flag to True.
            self.data["tf"]["tf_ok"] = True

        except Exception as e:

            # Display error.
            print(f"Robot is stationary.  Assignment is asking for dynamic information.  Please move the bot and try again.")

    def odom_callback(self, sample):

        def quaternion_to_yaw(quaternion):
            """
            Convert a quaternion to yaw (rotation around the Z-axis).
            
            :param quaternion: A tuple (x, y, z, w) representing the quaternion
            :return: Yaw angle in radians
            """
            x, y, z, w = quaternion
            yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            return yaw

        try:
            # Deserialize the message
            msg = deserialize_message(sample.payload.to_bytes(), Odometry)

            # Calculate yaw from quaternion
            # Extract quaternion (x, y, z, w) from orientation
            orientation = msg.pose.pose.orientation
            quaternion = (orientation.x, orientation.y, orientation.z, orientation.w)
            yaw = quaternion_to_yaw(quaternion)

            # Store the odometry data
            self.data["odometry"]["topic"] = "/odom"
            self.data["odometry"]["frame_id"] = msg.header.frame_id
            self.data["odometry"]["x"] = msg.pose.pose.position.x
            self.data["odometry"]["y"] = msg.pose.pose.position.y
            self.data["odometry"]["yaw"] = yaw
            self.data["odometry"]["vx"] = msg.twist.twist.linear.x
            self.data["odometry"]["vy"] = msg.twist.twist.linear.y
            self.data["odometry"]["wz"] = msg.twist.twist.linear.z

        except Exception as e:
            print(f"Error in odometry callback: {e}")

        
    def run(self):
        try:
            while True:
                time.sleep(0.1)  # allow callbacks to run

        except KeyboardInterrupt:
            print("Shutting down monitor...", flush=True)

        finally:

            self.sub_tf.undeclare()
            self.sub_robot_state.undeclare()
            self.sub_slam.undeclare()

            self.session.close()
            print("Monitor stopped.", flush=True)


if __name__ == "__main__":
    pipeline = semantic_pipeline(table, show_camera=False)
    pipeline.run()