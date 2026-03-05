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

from datetime import datetime

from db_connect import *

class DetectionPipeline:
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

        # Subscribe to images.
        self.sub_images = self.session.declare_subscriber("camera/image_raw", self.img_callback)
        print("Subscribed to: tb/camera/image_raw", flush=True)

        # Subscribe to detections.
        self.sub_detections = self.session.declare_subscriber("tb/detections", self.detections_callback)
        print("Subscribed to tb/detections", flush=True)

        # Subscribe to detections with wildcard in topic
        self.sub_maze_detections = self.session.declare_subscriber("maze/**/detections/v1/*", self.write_detections)
        print("Subscribed to maze/**/detections/v1/*", flush=True)

        # Subscribe to odom.
        self.sub_robot_state = self.session.declare_subscriber("odom", self.odom_callback)
        print(f"Subscribed to: odom")

        # Subscribe to TF
        self.sub_TF = self.session.declare_subscriber("tf_static", self.tf_callback)
        print(f"Subscribed to: tf")

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

    # Image callback to store the image in queue
    def img_callback(self, sample):
        try:
            # Deserialize the message
            msg = deserialize_message(sample.payload.to_bytes(), Image)

            # Store the image data in the structure
            self.data["image"]["topic"] = "/camera/image_raw"
            self.data["image"]["stamp"] = {
                "sec": msg.header.stamp.sec,
                "nanosec": msg.header.stamp.nanosec
            }
            self.data["image"]["frame_id"] = msg.header.frame_id
            self.data["image"]["width"] = msg.width
            self.data["image"]["height"] = msg.height
            self.data["image"]["encoding"] = msg.encoding
            self.data["image"]["sha256"] = "hex"  # Placeholder for SHA256

            # Optional: convert image to numpy and display
            img_data = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
            img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)

            # Put the image in the queue (discards oldest if full)
            if self.image_queue.full():
                try:
                    self.image_queue.get_nowait()
                except:
                    pass

            self.image_queue.put(img_data)

            # Optional display if show_camera is True
            if self.show_camera:
                cv2.imshow("Camera", img_data)
                cv2.waitKey(1)

        except Exception as e:
            print(f"Error in image callback: {e}")

    # Detection callback pulls latest image from queue
    def detections_callback(self, sample):
        if self.image_queue.empty():

            return

        try:

            detections = json.loads(sample.payload.to_bytes())

        except Exception as e:

            print(f"Failed to parse detection: {e}", flush=True)
            return

        if not detections:

            return

        # Get nested detection info.
        detection = detections[0]

        # Get the latest image from the queue
        try:
            img = self.image_queue.get_nowait()

        except:

            print("Failed to get image from queue", flush=True)
            return

        # Store a unique detection id.
        self.data['detections']['det_id'] = str(uuid.uuid4())  # unique det ID

        # Store the class id from item.
        from coco_dict import coco_item_class_dict

        self.data['detections']['class_id'] = coco_item_class_dict[detection['class']]

        # Store class name.
        self.data['detections']['class_name'] = detection['class']

        # Store confidence.
        self.data['detections']['confidence'] = detection['confidence']

        # Store bbox.
        self.data['detections']['bbox_xyxy'] = detection['bbox']  

        # Report detection.
        print(f"{self.data['detections']}", flush=True)
        
        # Publish data to Zenoh.
        """Serialize the data to JSON and publish it to Zenoh."""
        try:

            # Define the topic dynamically (e.g., maze/{robot_id}/detections/v1/{event_id})
            topic = f"maze/{self.data['robot_id']}/detections/v1/{self.data['event_id']}"

            # Serialize the data to JSON
            serialized_data = json.dumps(self.data)

            # Publish to Zenoh topic
            self.session.put(topic, serialized_data.encode())

        except Exception as e:
            print(f"Failed to publish to Zenoh: {e}") 
    
    def write_detections(self, sample):
    # Write json to db.

        data = json.loads(sample.payload.to_bytes())

        # Insert the data into PostgreSQL
        try:
            with psycopg.connect(
                f"dbname={dbname} user={user} password={password} host={host} port={port}"
            ) as conn:
                cursor = conn.cursor()

                # Prepare the INSERT query for the detection_events table
                insert_event_query = """
                    INSERT INTO detection_events (
                        event_id, run_id, robot_id, sequence, stamp,
                        image_frame_id, image_sha256, width, height, encoding,
                        x, y, yaw, vx, vy, wz,
                        tf_ok, t_base_camera, raw_event
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (event_id) DO NOTHING;
                """

                # Extract relevant data from the input JSON
                event_id = data["event_id"]
                run_id = data["run_id"]
                robot_id = data["robot_id"]
                sequence = data["sequence"]

                # Image-related data
                image_frame_id = data["image"]["frame_id"]
                image_sha256 = data["image"]["sha256"]
                width = data["image"]["width"]
                height = data["image"]["height"]
                encoding = data["image"]["encoding"]
                stamp = data["image"]["stamp"]

                # Convert stamp to datetime with timezone awareness
                stamp = datetime.fromtimestamp(
                    data["image"]["stamp"]["sec"] + data["image"]["stamp"]["nanosec"] * 1e-9,
                    timezone.utc
                )
                # Odometry data
                odometry = data["odometry"]

                # Round off floats.
                x = round(odometry["x"])
                y = round(odometry["y"])
                yaw = round(odometry["yaw"])
                vx = round(odometry["vx"])
                vy = round(odometry["vy"])
                wz = round(odometry["wz"])

                # Transform data
                tf_ok = data["tf"]["tf_ok"]
                t_base_camera = data["tf"]["t_base_camera"]

                # Prepare raw_event as JSONB
                raw_event = json.dumps(data)  # Convert the entire input JSON to a string
               
                # Execute the query to insert the event data
                cursor.execute(insert_event_query, (
                    event_id, run_id, robot_id, sequence, stamp,
                    image_frame_id, image_sha256, width, height, encoding,
                    x, y, yaw, vx, vy, wz,
                    tf_ok, t_base_camera, raw_event
                ))
                
                # Get detections.
                detection = data["detections"]
     
                det_id = detection["det_id"]
                class_id = detection["class_id"]
                class_name = detection["class_name"]
                confidence = detection["confidence"]
                x1, y1, x2, y2 = detection["bbox_xyxy"]

                # Round off floats.
                confidence = round(confidence, 3)
                x1 = round(x1)
                y1 = round(y1)
                x2 = round(x2)
                y2 = round(y2)

                
                # # Insert detection data into the detections table
                # cursor.execute(insert_detection_query, (
                #     event_id, det_id, class_id, class_name, confidence, x1, y1, x2, y2
                # ))

                # Only write in rows of new data.  Omit already recorded detections.
                insert_detection_query = """                                                                                                                                                                
                    INSERT INTO detections (                                                                                                                                                                
                        event_id, det_id, class_id, class_name, confidence, x1, y1, x2, y2                                                                                                                  
                    )           
                    SELECT %s, %s, %s, %s, %s, %s, %s, %s, %s
                    WHERE NOT EXISTS (
                        SELECT 1 FROM detections
                        WHERE class_name = %s
                            AND confidence = %s
                            AND x1 = %s AND y1 = %s AND x2 = %s AND y2 = %s
                    );
                """

                 # Insert detection data into the detections table
                cursor.execute(insert_detection_query, (
                    event_id, det_id, class_id, class_name, confidence, x1, y1, x2, y2,  # INSERT values
                    class_name, confidence, x1, y1, x2, y2                                # WHERE NOT EXISTS values
                ))

                
                # Commit the transaction to save the data in the database
                conn.commit()
                #print("Data inserted successfully!")

        except Exception as e:
            print(f"Error inserting data: {e}")

    
    def run(self):
        try:
            while True:
                time.sleep(0.1)  # allow callbacks to run

        except KeyboardInterrupt:
            print("Shutting down monitor...", flush=True)

        finally:

            self.sub_images.undeclare()
            self.sub_detections.undeclare()
            self.sub_maze_detections.undeclare()
            self.sub_robot_state.undeclare()
            self.sub_TF.undeclare()

            self.session.close()
            print("Monitor stopped.", flush=True)


if __name__ == "__main__":
    pipeline = DetectionPipeline(table, show_camera=False)
    pipeline.run()