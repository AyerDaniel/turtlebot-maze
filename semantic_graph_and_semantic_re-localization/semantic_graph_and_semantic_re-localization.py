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

from psycopg.rows import dict_row
import clip
from PIL import Image
import torch
from datetime import datetime, timezone
from pgvector.psycopg import register_vector 

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances  

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

def detect_objects():

    '''
        
        This function assumes zenoh-router, zenoh-bridge, dectector, demo-world-enhanced containers are running.

    '''

    # Init rclpy.
    rclpy.init() 

    # Instantiate pipeline to utilize Zenoh and such to record detections from ROS2 and Gazebo.
    pipeline = semantic_pipeline(table, show_camera=False)

    # Run pipeline.
    pipeline.run()

def create_embeddings():

    '''

        This function assumes the postgres container is running.
        This container should have the tables holding the detections and detection events.

    '''
    try:
        
        # Set device and load model to create embeddings.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)

        with psycopg.connect(
            f"dbname={dbname} user={user} password={password} host={host} port={port}",
            row_factory=dict_row
        ) as conn:
            
            # Create a cursor. 
            cursor = conn.cursor()

            # Prepare query sql.
            query_sql = """
                SELECT det_pk, class_name, x1, y1, x2, y2, image_raw FROM detections LEFT JOIN detection_events ON detections.event_id = detection_events.event_id
            """
            
            # Execute query.
            cursor.execute(query_sql)

            # Get all detections.
            detections = cursor.fetchall()

            # Create embeddings for each row result.  Each row is a detection.
            for row in detections:
                
                # Store each returned field as a variable.
                det_pk = row['det_pk']                                                                                                                                                                      
                # event_id = row['event_id']
                # det_id = row['det_id']                                                                                                                                                                      
                # class_id = row['class_id']
                class_name = row['class_name']                                                                                                                                                              
                # confidence = row['confidence']
                x1 = row['x1']                                                                                                                                                                              
                y1 = row['y1']                                                                                                                                                                              
                x2 = row['x2']                                                                                                                                                                              
                y2 = row['y2']                                                                                                                                                                              
                # timestamp = row['timestamp']
                # run_id = row['run_id']
                # robot_id = row['robot_id']                                                                                                                                                                  
                # sequence = row['sequence']
                # time_in_run = row['time_in_run']                                                                                                                                                            
                # image_frame_id = row['image_frame_id']
                # image_sha256 = row['image_sha256']                                                                                                                                                          
                # width = row['width']                                                                                                                                                                        
                # height = row['height']                                                                                                                                                                      
                # encoding = row['encoding']                                                                                                                                                                  
                # x = row['x']                                                                                                                                                                                
                # y = row['y']    
                # yaw = row['yaw']
                # vx = row['vx']                                                                                                                                                                              
                # vy = row['vy']                                                                                                                                                                              
                # wz = row['wz']                                                                                                                                                                              
                # tf_ok = row['tf_ok']                                                                                                                                                                        
                # t_base_camera = row['t_base_camera']
                # raw_event = row['raw_event']
                image_raw = row['image_raw']

                # Read in image.  Expected encoding: 'rgb8, bytes: 19437' output from ROS2.
                nparr = np.frombuffer(image_raw, np.uint8)                                                                                                                                                  
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)                                                                                                                                                 
                
                # Crop image to bbox coords.
                cropped = img[int(y1):int(y2), int(x1):int(x2)] 

                # Create PIL input for CLIP.
                pil_image = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

                # Preprocess for CLIP                                                                                                                                                                   
                image_tensor = preprocess(pil_image).unsqueeze(0).to(device)                                                                                                                            
                
                # Get embeddings.
                with torch.no_grad():                                                                                                                                                                   
                    image_features = model.encode_image(image_tensor)                                                                                                                                   
                
                # Create sql to write to new table.
                write_embeddings_sql = '''
                    INSERT INTO detection_embeddings (det_pk, model, embedding) VALUES (%s, %s, %s)
                    ON CONFLICT (det_pk) DO NOTHING
                '''

                # Move embedding to cpu.
                image = image_features.cpu()

                # Convert to numpy array.
                image = image.numpy()
                
                # Flatten, is [1, 512] for some reason.
                image = image.flatten()
                
                # Convert to list for psycopg.
                image = image.tolist()

                try:

                    # Stage write to table.
                    cursor.execute(write_embeddings_sql, (row['det_pk'], row['class_name'], image))

                    # Commit to db.
                    conn.commit()

                except Exception as e:
                    # Report problems.
                    print(e)


    except Exception as e:
        print(f"Error inserting data: {e}")

def do_age():

    # Connect to db.
    try:
            
        with psycopg.connect(
                f"dbname={dbname} user={user} password={password} host={host} port={port}",
                row_factory=dict_row
            ) as conn:
            
            # Create cursor.
            cursor = conn.cursor()

            # Prepare query sql.
            query_sql = """
                SELECT * FROM detections LEFT JOIN detection_events ON detections.event_id = detection_events.event_id
            """

            # List to store x and y values.
            coords = []
            det_pks = []

            # Get detections.
            cursor.execute(query_sql)
            detections = cursor.fetchall()

            for row in detections:
                
                # Store det_pk.
                det_pks.append(row['det_pk'])

                # Store detection's coordinates.
                coords.append((row['x'], row['y']))
   
            # Run DBSCAN.  Set min_samples to 1 because all objects occur at least once.
            dbscan = DBSCAN(eps=0.06, min_samples=1, metric='cosine')
            labels = dbscan.fit_predict(coords)   

            # Display results.
            for det_pk, label in zip(det_pks, labels):                                                                                                                                                  
                print(f"det_pk: {det_pk}, cluster: {label}")


    except Exception as e:
        
        print(f"psycopg connection to db threw exception: {e}")

if __name__ == "__main__":
    

    detect_objects()
    #create_embeddings()
    #do_age()

    


