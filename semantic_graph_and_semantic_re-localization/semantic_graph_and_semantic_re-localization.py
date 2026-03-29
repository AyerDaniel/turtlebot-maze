#!/usr/bin/env python3
"""
Object-detection pipeline from TurtleBot3 simulation through Zenoh into PostgreSQL
using an image queue to pair images with detections.
"""

import threading

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
import hashlib
import pandas as pd

from psycopg.rows import dict_row
import clip
from PIL import Image as PILImage
import torch
from datetime import datetime, timezone
from pgvector.psycopg import register_vector 

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances  

from scipy.spatial import cKDTree

# Import packages for deserializing Zenoh output from ROS2 sensor_msgs/msg/Image.
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose

from ultralytics import YOLO

from datetime import datetime

from db_connect import *

class Turtlebot:
    def __init__(self, table, queue_size=10, show_camera=True):
        self.table = table
        self.show_camera = show_camera
        self.image_queue = Queue(maxsize=queue_size)

        # Initialize the data structure
        self.data = {                                                                                                                                                                               
            "schema": "maze.detection.v1",
            "event_id": str(uuid.uuid4()),
            "run_id": str(uuid.uuid4()),                                                                                                                                                            
            "robot_id": "tb3_sim",
            "sequence": 0,                                                                                                                                                                          
            "image": {  
                "topic": None,
                "stamp": None,                                                                                                                                                                      
                "frame_id": None,
                "width": None,                                                                                                                                                                      
                "height": None,
                "encoding": None,
                "sha256": None,
                "current_image": None,                                                                                                                                                              
            },
            "odometry": {                                                                                                                                                                           
                "topic": None,
                "frame_id": None,
                "x": 0.0, "y": 0.0, "yaw": 0.0,                                                                                                                                                     
                "vx": 0.0, "vy": 0.0, "wz": 0.0,
            },                                                                                                                                                                                      
            "tf": {     
                "base_frame": None,                                                                                                                                                                 
                "camera_frame": None,                                                                                                                                                               
                "t_base_camera": None,
                "tf_ok": False,                                                                                                                                                                     
            },          
            "detections": {                                                                                                                                                                         
                "det_id": None,
                "class_id": None,
                "class_name": None,                                                                                                                                                                 
                "confidence": None,
                "bbox_xyxy": None,                                                                                                                                                                  
            },          
            "pose": {
                "timestamp": None,
            },                                                                                                                                                                                      
            "slam": {
                "timestamp": None,                                                                                                                                                                  
            },          
            "rotational": None,
            "batch_cropped_detections": []                                                                                                                                                          
        }


        # Zenoh connection
        conf = zenoh.Config()
        conf.insert_json5("connect/endpoints", '["tcp/localhost:7447"]')

        # Run YOLOv8 inference on the frame to produce bounding boxes
        self.yolo_model = YOLO('yolov8n.pt') 

        # Set device and model to create embeddings.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        
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
        # self.sub_maze_detections = self.session.declare_subscriber("maze/**/detections/v1/*", self.write_detections)
        # print("Subscribed to maze/**/detections/v1/*", flush=True)

        # Subscribe to odom.
        self.sub_robot_state = self.session.declare_subscriber("odom", self.odom_callback)
        print(f"Subscribed to: odom", flush=True)

        # Subscribe to TF
        self.sub_TF = self.session.declare_subscriber("tf_static", self.tf_callback)
        print(f"Subscribed to: tf", flush=True)

        # Subscribe to SLAM.
        self.sub_slam = self.session.declare_subscriber("tb/slam/pose", self.slam_pose_callback)
        print(f"Subscribed to slam/pose", flush=True)

        self.sub_slam = self.session.declare_subscriber("tb/slam/status", self.slam_status_callback)
        print(f"Subscribed to slam/status", flush=True)

    # End of zenoh subscribing.

    def write_to_graph(self, conn, json_, graph='maze'):
        # This method writes detections to the graph for later use.

        def label_exists(cursor, graph, label):                                                                                                                                                     
            # This function checks for existing nodes and edges.
            cursor.execute("""                                                                                                                                                                      
                SELECT count(*) FROM ag_catalog.ag_label                                                                                                                                            
                WHERE graph = (SELECT graphid FROM ag_catalog.ag_graph WHERE name = %s)                                                                                                             
                AND name = %s
            """, (graph, label))                                                                                                                                                                    
            return cursor.fetchone()[0] > 0
        
        # Get cursor.
        cursor = conn.cursor()

        # Load Apache AGE.
        cursor.execute("CREATE EXTENSION IF NOT EXISTS age")
        cursor.execute("LOAD 'age'")
        cursor.execute("SET search_path = ag_catalog, \"$user\", public")

        # Create graph if needed.
        # Query for existing graph.
        cursor.execute("SELECT count(*) FROM ag_catalog.ag_graph WHERE name = %s", (graph,))

        # Create graph if needed.                                                                                                                                                       
        if cursor.fetchone()[0] == 0:

            cursor.execute("SELECT * FROM ag_catalog.create_graph(%s)", (graph,))  
        
        # Create list of nodes.
        nodes = ['Run', 'Keyframe', 'Pose', 'Place', 'Object', 'Observation']      

        # Create list of edge types.
        edges = ['HAS_KEYFRAME', 'HAS_POSE', 'HAS_OBSERVATION', 'CORRESPONDS_TO', 'LOCATED_IN', 'ADJACENT_TO']                                                                                    

        # Create node labels if needed.                                                                                                                                                                                    
        for node in nodes:                                                                                                                                                                       
            if not label_exists(cursor, 'maze', node):                                                                                                                                             
                try:

                    cursor.execute("SELECT create_vlabel('maze', %s)", (node,))

                except Exception as e:

                    print(f"Exception {e}")

        # Create edge labels if needed.                                                                                                                                                                         
        for edge in edges:
            if not label_exists(cursor, 'maze', edge):                                                                                                                                             
                try:

                    cursor.execute("SELECT create_elabel('maze', %s)", (edge,))
                
                except Exception as e:
                    
                    print(f"Exception {e}")

        # Set variables for readability.                                                                                                                                                      
        run_id = self.data['run_id']                                                                                                                                                          
        kf_id  = json_['keyframe_id']                                                                                                                                                         
        ts     = json_['timestamp']                                                                                                                                                           
        x      = json_['map_x']                                                                                                                                                               
        y      = json_['map_y']                                                                                                                                                               
        yaw    = json_['map_yaw'] 


        # Create run_id and timestamp.
        try:
            params = json.dumps({"run_id": run_id, "started_at": datetime.now().isoformat()})                                                                                                           
            cursor.execute("""
                    SELECT * FROM cypher('maze', $$                                                                                                                                                         
                        MATCH (r:Run {run_id: $run_id})
                        RETURN r                                                                                                                                                                            
                    $$, %s::agtype) AS (r agtype)
                """, (params,))

            if cursor.fetchone() is None:
                # Run node doesn't exist yet — create it with started_at.
                params = json.dumps({"run_id": run_id, "started_at": datetime.now().isoformat()})                                                                                                   
                cursor.execute("""                                                                                                                                                                  
                    SELECT * FROM cypher('maze', $$                                                                                                                                                 
                        CREATE (r:Run {run_id: $run_id, started_at: $started_at})                                                                                                                   
                        RETURN r
                    $$, %s::agtype) AS (r agtype)
                """, (params,))
        
        except Exception as e:                                                                                                                                                                      
            print(f"Exception creating Run node: {e}")
            conn.rollback() 

        # Set variables for readability.
        params = json.dumps({
              "run_id": run_id,
              "kf_id":  kf_id,
              "ts":     ts,
              "x":      x,
              "y":      y,
              "yaw":    yaw,
          })

        try:
            cursor.execute("""
                SELECT * FROM cypher('maze', $$
                    MATCH (r:Run {run_id: $run_id})
                    CREATE (kf:Keyframe {keyframe_id: $kf_id, timestamp: $ts})
                    CREATE (p:Pose {x: $x, y: $y, yaw: $yaw, dbscan_group: -1})
                    CREATE (r)-[:HAS_KEYFRAME]->(kf)
                    CREATE (kf)-[:HAS_POSE]->(p)
                    RETURN kf
                $$, %s::agtype) AS (kf agtype)
            """, (params,))
        
        except Exception as e:

            print(f"Exception {e}")

        # store detections.     
        for det in json_.get('detections', []):
            params = json.dumps({
                  "kf_id":      kf_id,
                  "class_name": det['class'],
                  "confidence": det['confidence'],
                  "bbox":       json.dumps(det['bbox']),   # store as JSON string
                  "emb_model":  det['embedding_model'],
                  "embedding":  det['embedding'],
              })

            try:
                cursor.execute("""
                        SELECT * FROM cypher('maze', $$
                            MATCH (kf:Keyframe {keyframe_id: $kf_id})
                            MERGE (obj:Object {class_name: $class_name})
                            CREATE (obs:Observation {
                                confidence:      $confidence,
                                bbox:            $bbox,
                                embedding_model: $emb_model,
                                embedding:       $embedding
                            })
                            CREATE (kf)-[:HAS_OBSERVATION]->(obs)
                            CREATE (obs)-[:CORRESPONDS_TO]->(obj)
                            RETURN obs
                        $$, %s::agtype) AS (obs agtype)
                    """, (params,))

            except Exception as e:

                print(f"Exception {e}")

        # Commit changes.
        try:
            conn.commit()

        except Exception as e:

            print(f"Exception: {e}")


    # End of write_to_graph().

    def rotation_delta_rad(self, mat_prev, mat_curr):  

        # Extract 3x3 matrix of rotational values from SLAM Pose Zenoh Topic.                                                                                                                                               
        R_prev = mat_prev[:3, :3]                                                                                                                                                               
        R_curr = mat_curr[:3, :3]                                

        # Compute rotation between matrices.  For rotational matrix transpose is inverse as per documentation.                                                                                                                               
        R_rel = R_curr @ R_prev.T

        # Get cos of angle.
        cos_angle = np.clip((np.trace(R_rel) - 1) / 2, -1, 1)                                                                                                                                   
        
        return np.arccos(cos_angle)        

    # End of rotation_delta_rad().
    def write_embeddings(self, conn, detections):                                                                                                                                               
      cursor = conn.cursor()                                                                                                                                                                  
      register_vector(conn)                                                                                                                                                                   
                                                                                                                                                                                              
      for det in detections:                                                                                                                                                                
          try:
              cursor.execute("""
                  INSERT INTO detection_embeddings (keyframe_id, model, embedding)
                  VALUES (%s, %s, %s)
                  ON CONFLICT DO NOTHING                                                                                                                                                      
              """, (
                  det['keyframe_id'],                                                                                                                                                         
                  det['embedding_model'],                                                                                                                                                   
                  det['embedding']
              ))
          except Exception as e:
              print(f"Failed to write embedding: {e}", flush=True)
                                                                                                                                                                                              
      conn.commit()

    # End of write_embeddings().
    
    def slam_pose_callback(self, sample):

        # Get JSON from Zenoh Topic.
        data = json.loads(bytes(sample.payload))
            
         # Handle first-run case where timestamps may not be initialized                                                                                                                         
        if self.data.get('pose', {}).get('timestamp') is None:                                                                                                                                  
            self.data.setdefault('pose', {})['timestamp'] = time.time()                                                                                                      
            return     
                                                                                                                                                                     
        # Set time window to ignore new images in ms.
        del_t = 100  # In miliseconds.

        # Rate limit check - if less than 100 ms since the last frame was considered, skip immediately #                                                                                                                                                                   
        if self.data['slam']['timestamp'] - self.data['pose']['timestamp'] <= 0.01 * del_t:
            # Not enough time has elapsed.
            return

        # Cache the latest robot pose from the odometry subscriber stored in internal data structure.
        x = self.data["odometry"]["x"]
        y = self.data["odometry"]["y"]
        yaw = self.data["odometry"]["yaw"]

        '''
            Keyframe gate - compare the current pose against the pose of the last accepted keyframe:
                If the robot has moved less than 0.5 m AND rotated less than 15 degrees, discard the frame (no inference)
                If either threshold is exceeded, this frame is a keyframe - proceed to step 4

        '''

        # Convert Zenoh SLAM Pose topic into rotational and positional vectors.
        mat = np.array(data['pose']).reshape(3, 4) 
        rotational = mat[:3, :3]
        translational = mat[:, 3]

        # First-run guard for rotational matrix.
        if self.data.get('rotational') is None:                                                                                                                                                     
            self.data['rotational'] = rotational
            return   

        # Set euclidean threshold.
        del_euc = 0.5  # In m for Gazebo assuming one cube is 1m X 1m x 1m.

        # Set rotational threshold. Convert degrees into rads.
        rot_thresh = np.radians(15)

        # Calculate euclidean_dist.
        euclidean_dist = np.sqrt((x - translational[0])**2 + (y - translational[1])**2)

        # Calculate the rotational sweep.  Returns rads.
        rotation = self.rotation_delta_rad(self.data['rotational'], rotational)

        # Check for distance and rotation thresholds.
        if euclidean_dist <= del_euc and rotation <= rot_thresh:
            # Neither threshold has been reached.  Discard.
            return
        
        ####  All checks passed ###           

        # Deserialize the CDR-encoded image and decode it to a numpy array.

        '''
            img_callback does this and stores image in internal data structure:
            self.data['image']['current_image']

        '''

        # Create current image variable for readability.
        curr_image = self.data['image']['current_image']

        if curr_image is None:                                                                                                                                                                      
            print("No image available yet, skipping frame.", flush=True)
            return 

        # Run YOLO inference on a single image.
        results = self.yolo_model(curr_image)

        keyframe_id = str(uuid.uuid4())  # one ID for this frame
        detections = [] 

        for result in results:
            
            # See if there is a detection.
            if len(result.boxes) == 0:
                # Nothing detected.

                # Create PIL input for CLIP.  We're encoding the entire image.
                pil_image = PILImage.fromarray(cv2.cvtColor(curr_image, cv2.COLOR_BGR2RGB))

                # Preprocess for CLIP                                                                                                                                                                   
                image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)

                # Get embeddings.
                with torch.no_grad():                                 

                    # Get embeddings.
                    image_features = self.clip_model.encode_image(image_tensor) 

                    # Apply L2 norm.
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True) 
                
                # Create JSON envelope.
                json_envelope = {
                    "event_id": str(uuid.uuid4()),                                                                                                                                                          
                    "run_id": self.data["run_id"],
                    "robot_id": self.data["robot_id"],
                    "sequence": self.data["sequence"],
                    "odometry": self.data["odometry"],                                                                                                                                                      
                    "tf": self.data["tf"],            
                    "keyframe_id": keyframe_id,                                                                                                                                                             
                    "timestamp": datetime.now().isoformat(),
                    "map_x": self.data["odometry"]["x"],    
                    "map_y": self.data["odometry"]["y"],                                                                                                                                                    
                    "map_yaw": self.data["odometry"]["yaw"],
                    "detections": []                                                                                                                                                               
                }        
                
                # Write to graph.
                with psycopg.connect(
                    f"dbname={dbname} user={user} password={password} host={host} port={port}"
                    ) as conn:                                                                                                                                                                                  
                    self.write_to_graph(conn, json_envelope, graph='maze')

                    # Store embeddings.
                    self.write_embeddings(conn, detections)

                # Update pose timestamp to now for rate limiting next call                                                                                                                          
                self.data['pose']['timestamp'] = self.data['slam']['timestamp']

                # Update rotational matrix for next keyframe comparison.
                self.data['rotational'] = rotational
                
            
            else:

                # Something found.

                # Get box coords of detected object.
                for box in result.boxes:

                    # Get coords of box.
                    x1, y1, x2, y2 = box.xyxy[0].tolist()          

                    # Get most probable class of box.                                                                                                                                         
                    cls = int(box.cls[0].item())     

                    # Get confidence.    
                    confidence = float(box.conf[0].item())           

                    # Get name of class.                                                                                                                                       
                    class_name = result.names[cls]   

                    # Crop image.
                    cropped = curr_image[int(y1):int(y2), int(x1):int(x2)]

                    '''
                        The assignment says to:
                        Batch all crops through the CLIP encoder (ViT-B/32) to produce 512-dim L2-normalized embeddings

                        However, I am going to create the embeds as the image comes.  
                        The business logic to batch the clips 
                        remains to satisfy the assignment requirements.

                    '''

                    # Add image to batch.
                    self.data['batch_cropped_detections'].append(cropped)

                    # Create PIL input for CLIP.
                    pil_image = PILImage.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

                    # Preprocess for CLIP                                                                                                                                                                   
                    image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)                                                                                                                            
                    
                    # Get embeddings.
                    with torch.no_grad():                                 

                        # Get embeddings.
                        image_features = self.clip_model.encode_image(image_tensor) 

                        # Apply L2 norm.
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  

                    detections.append({                                                                                                                                                                 
                        'class': class_name,
                        'confidence': confidence,                                                                                                                                                       
                        'bbox': (x1, y1, x2, y2),
                        'embedding': image_features.squeeze(0).tolist(),
                        'embedding_dim': 512,                                                                                                                                                           
                        'embedding_model': 'CLIP: ViT-B/32',
                        'keyframe_id': keyframe_id
                    })  

                    '''
                        Publish a JSON envelope to tb/detections containing:
                            keyframe_id - monotonically increasing integer
                            timestamp - wall clock time
                            map_x, map_y, map_yaw - robot pose in map frame
                            detections - array of \\{class, confidence, bbox, embedding, embedding_dim, embedding_model\\}

                    '''
                
                # If detections publish.
                if detections:
                    # Create JSON envelope.
                    json_envelope = {
                        "event_id": str(uuid.uuid4()),                                                                                                                                                          
                        "run_id": self.data["run_id"],
                        "robot_id": self.data["robot_id"],
                        "sequence": self.data["sequence"],
                        "odometry": self.data["odometry"],                                                                                                                                                      
                        "tf": self.data["tf"],            
                        "keyframe_id": keyframe_id,                                                                                                                                                             
                        "timestamp": datetime.now().isoformat(),
                        "map_x": self.data["odometry"]["x"],    
                        "map_y": self.data["odometry"]["y"],                                                                                                                                                    
                        "map_yaw": self.data["odometry"]["yaw"],
                        "detections": detections                                                                                                                                                               
                    }    

                    # Set topic.
                    topic = "tb/detections"

                    # Serialize the data to JSON
                    serialized_data = json.dumps(json_envelope)

                    # Publish topic to Zenoh.
                    self.session.put(topic, serialized_data.encode())

                # Write to graph.
                with psycopg.connect(
                    f"dbname={dbname} user={user} password={password} host={host} port={port}"
                    ) as conn:                                                                                                                                                                                  
                    self.write_to_graph(conn, json_envelope, graph='maze')

            # Update pose timestamp to now for rate limiting next call                                                                                                                          
            self.data['pose']['timestamp'] = self.data['slam']['timestamp']

            # Update rotational matrix for next keyframe comparison.
            self.data['rotational'] = rotational  

        ##  DBSCAN ##
            
            '''
                Online DBSCAN as described will fail.  
                We sample the environemnt every 0.5m, but have a grouping radius of 1.5m.
                Therefore, online DBSCAN will simple grow one group out infinitely.
                I am switching to offline DBSCAN for group clustering assignments.
                This will provide information for the remapping function.

            ''' 
    # End of slam_pose_callback().

    def slam_status_callback(self, sample):

        # Process JSON.
        data = json.loads(bytes(sample.payload))

        # Get timestamp from SLAM status.
        self.data['slam']['timestamp'] = data['timestamp']
    
    # End of slam_status_callback.

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
            print(f"Robot is stationary.  Assignment is asking for dynamic information.  Please move the bot and try again.\nError thrown {e}")

    # End of tf_callback()

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

    # End of odom_callback().

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
            self.data["image"]["sha256"] = hashlib.sha256(msg.data.tobytes()).hexdigest()

            # Optional: convert image to numpy and display
            img_data = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
            img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)

            # Save image to self.
            self.data['image']['current_image'] = img_data

            # Put the image in the queue (discards oldest if full)
            if self.image_queue.full():
                try:
                    self.image_queue.get_nowait()
                except Exception as e:
                    # Report error.
                    print(f"Exception thrown: {e}")

            self.image_queue.put(img_data)

            # Optional display if show_camera is True
            if self.show_camera:
                cv2.imshow("Camera", img_data)
                cv2.waitKey(1)

        except Exception as e:
            print(f"Error in image callback: {e}")

    # End of img_callback().

    def to_serializable(self, obj):                                                                                                                                                                   
        if isinstance(obj, np.ndarray):                                                                                                                                                         
            return obj.tolist()                                                                                                                                                                 
        elif isinstance(obj, dict):                                                                                                                                                             
            return {k: self.to_serializable(v) for k, v in obj.items()}                                                                                                                              
        elif isinstance(obj, (list, tuple)):
            return [self.to_serializable(i) for i in obj]                                                                                                                                            
        elif isinstance(obj, np.integer):
            return int(obj)                                                                                                                                                                     
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, bytes):
            return obj.decode('utf-8')                                                                                                                                                          
        return obj

    # End of to_serializable.

    # Detection callback pulls latest image from queue
    def detections_callback(self, sample):
 
        if not sample:

            return
 
        # Store a unique detection id.
        event_id = str(uuid.uuid4())
                
        # Publish data to Zenoh.
        """Serialize the data to JSON and publish it to Zenoh."""
        try:

            # Define the topic dynamically (e.g., maze/{robot_id}/detections/v1/{event_id})
            topic = f"maze/{self.data['robot_id']}/detections/v1/{event_id}"

            # Publish to Zenoh topic
            self.session.put(topic, sample.payload.to_bytes())

        except Exception as e:
            print(f"Failed to publish to Zenoh: {e}") 
    
    # End of detections_callback().

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
                        event_id, run_id, robot_id, sequence, time_in_run,
                        image_frame_id, image_sha256, width, height, encoding,
                        x, y, yaw, vx, vy, wz,
                        tf_ok, t_base_camera, raw_event, image_raw
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                #time_in_run = data["image"]["stamp"]

                # Convert stamp to datetime with timezone awareness
                time_in_run = datetime.fromtimestamp(
                    data["image"]["stamp"]["sec"] + data["image"]["stamp"]["nanosec"] * 1e-9,
                    timezone.utc
                )

                # Odometry data
                odometry = data["odometry"]

                # Round off floats.
                x = round(odometry["x"], 4)
                y = round(odometry["y"], 4)
                yaw = round(odometry["yaw"], 4)
                vx = round(odometry["vx"], 4)
                vy = round(odometry["vy"], 4)
                wz = round(odometry["wz"], 4)

                # Transform data
                tf_ok = data["tf"]["tf_ok"]
                t_base_camera = data["tf"]["t_base_camera"]


                # Prepare raw_event as JSONB
                raw_event = json.dumps(data)  # Convert the entire input JSON to a string
               
               # Prepare image.
                success, encoded = cv2.imencode('.jpg', self.data['image']['current_image'])                                                                                                                                 
                img_bytes = encoded.tobytes() if success else None 

                # Execute the query to insert the event data
                try:
                    cursor.execute(insert_event_query, (
                        event_id, run_id, robot_id, sequence, time_in_run,
                        image_frame_id, image_sha256, width, height, encoding,
                        x, y, yaw, vx, vy, wz,
                        tf_ok, t_base_camera, raw_event, img_bytes
                    ))
                except Exception as e:
                    print(f"Failed to write to detection_events.  Error: {e}")
                    return
                
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
                    INSERT INTO detections (event_id, det_id, class_id, class_name, confidence, x1, y1, x2, y2)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (x1, y1, x2, y2)
                        DO UPDATE SET
                            det_id     = EXCLUDED.det_id,
                            class_id   = EXCLUDED.class_id,
                            class_name = EXCLUDED.class_name,
                            confidence = EXCLUDED.confidence
                        WHERE EXCLUDED.confidence > detections.confidence;
                """

                # Try to write to 'detections'.  If not, then duplicate entry and don't write to 'detection_events'.
                try:
                
                    # Insert detection data into the detections table
                    cursor.execute(insert_detection_query, (
                        event_id, det_id, class_id, class_name, confidence, x1, y1, x2, y2
                    ))
                
                    # If didn't write with no exception don't write to detection_events.
                    if cursor.rowcount == 0:                                                                                                                                                                
                        print(f"Detection skipped: existing bbox {x1,y1,x2,y2} has higher confidence.", flush=True)                                                                                         
                        return 
                    
                except psycopg.errors.UniqueViolation:                                                                                                                                                      
                    print(f"Uniqueness conflict on detections. Aborting.", flush=True)                                                                                                                      
                    return      
                
                except Exception as e:                                                                                                                                                                      
                    print(f"Failed to write to detections. Aborting. Error: {e}", flush=True)                                                                                                               
                    return 

                
                # Commit the transaction to save the data in the database
                conn.commit()
                #print("Data inserted successfully!")

        except Exception as e:
            print(f"Error inserting data: {e}")

    # End of write_detections().

    
    def run(self):
        try:
            while True:
                time.sleep(0.1)  # allow callbacks to run

        except KeyboardInterrupt:
            print("Shutting down monitor...", flush=True)

        finally:

            self.sub_images.undeclare()
            self.sub_detections.undeclare()
            # self.sub_maze_detections.undeclare()
            self.sub_robot_state.undeclare()
            self.sub_TF.undeclare()

            self.session.close()
            print("Monitor stopped.", flush=True)

def detect_objects(turtlebot):

    '''
        
        This function assumes zenoh-router, zenoh-bridge, dectector, demo-world-enhanced containers are running.

    '''

    # Init rclpy.
    rclpy.init() 

    # Run pipeline.
    turtlebot.run()


def dbscan():
        # This function performs dbscan on the sampled space.

        def ag(v):                                                                                                                                                                                  
            if v is None: return None                                                                                                                                                               
            s = str(v)                                                                                                                                                                              
            return s.strip('"')

        # End of ag().

        def parse_node(v):                                                                                                                                                                          
            s = str(v)                                                                                                                                                                              
            if '::' in s:                                                                                                                                                                           
                s = s.rsplit('::', 1)[0]                                                                                                                                                            
            return json.loads(s)

        # End parse_node()

        try:
            with psycopg.connect(
                f"dbname={dbname} user={user} password={password} host={host} port={port}"
                ) as conn:    

                    # Build cursor.
                    cursor = conn.cursor()

                    # Activate AGE.
                    cursor.execute("LOAD 'age'")                                                                                                                                                                   
                    cursor.execute("SET search_path = ag_catalog, \"$user\", public")

                    # All keyframes with pose coords                                                                                                                                                            
                    cursor.execute("""                                                                                                                                                                             
                        SELECT * FROM cypher('maze', $$                                                                                                                                                         
                            MATCH (kf:Keyframe)-[:HAS_POSE]->(p:Pose)                                                                                                                                           
                            RETURN kf, p.x, p.y, p.dbscan_group                                                                                                                                                                 
                        $$) AS (kf agtype, x agtype, y agtype, dbscan_group agtype)
                    """)       

                    all_kfs = {                                                                                                                                                                                 
                        (props := parse_node(row[0])['properties'])['keyframe_id']: {
                            **props,                                                                                                                                                                            
                            'x': float(ag(row[1])),
                            'y': float(ag(row[2])),                                                                                                                                                             
                            'dbscan_group': int(ag(row[3]))
                        }                                                                                                                                                                                       
                        for row in cursor.fetchall()
                    }      
                                                                                                                                                                                                                
                    # Keyframe IDs that have at least one detection                                                                                                                                             
                    cursor.execute("""
                        SELECT * FROM cypher('maze', $$                                                                                                                                                         
                            MATCH (kf:Keyframe)-[:HAS_OBSERVATION]->()
                            RETURN DISTINCT kf.keyframe_id                                                                                                                                                      
                        $$) AS (keyframe_id agtype)
                    """)                                   

                    # Get nodes to filter for detections.                                                                                                                                                     
                    det_ids = {ag(r[0]) for r in cursor.fetchall()}

                    # Get ids of nodes with detections.                                                                                                                                                                                
                    det_kfs = {kf_id: all_kfs[kf_id] for kf_id in det_ids if kf_id in all_kfs}

                    if not det_kfs:                                                                                                                                                                             
                        print("DBSCAN: no detection keyframes yet, skipping.", flush=True)                                                                                                                      
                        return    
                    
                    # Get spatial coordinates of detections.
                    det_coords = np.array([(props['x'], props['y']) for props in det_kfs.values()]) 

                    # DBSCAN coords for groupings.  I expect 4 as of 03282026
                    db_eps = 1.5
                    min_samples = 3
                    
                    
                        
                    db = DBSCAN(eps = db_eps, min_samples = min_samples, metric='euclidean').fit(det_coords)

                    # Assign clusters.
                    det_cluster = {kf_id: int(label) for kf_id, label in zip(det_kfs.keys(), db.labels_)}

                    # Build KD-tree from non-noise clustered detection keyframes.                                                                                                                       
                    clustered = [                                                                                                                                                                       
                        (kf_id, props['x'], props['y'], det_cluster[kf_id])                                                                                                                             
                        for kf_id, props in det_kfs.items()
                        if det_cluster[kf_id] >= 0
                    ]                                                   

                    if not clustered:                                                                                                                                                                           
                        print("DBSCAN: no clusters formed — all detections are noise. Try lowering min_samples.", flush=True)
                        return 
                                                                                                                                    
                    clustered_coords = np.array([(x, y) for _, x, y, _ in clustered])
                    clustered_labels = np.array([label for _, _, _, label in clustered])                                                                                                                
                    tree = cKDTree(clustered_coords)                                                                                                                                                    
            
                    # Assign noise detection keyframes to nearest valid cluster.                                                                                                                        
                    for kf_id, label in list(det_cluster.items()):
                        if label == -1:
                            _, idx = tree.query([det_kfs[kf_id]['x'], det_kfs[kf_id]['y']])
                            det_cluster[kf_id] = int(clustered_labels[idx])
                                                                                                                                                                                                        
                    # Assign non-detection keyframes to nearest valid cluster.
                    for kf_id, props in all_kfs.items():                                                                                                                                                
                        if kf_id not in det_cluster:                                                                                                                                                    
                            _, idx = tree.query([props['x'], props['y']])
                            det_cluster[kf_id] = int(clustered_labels[idx])                                                                                                                             
                            
                    # Write group assignments back to Pose nodes.                                                                                                                                       
                    for kf_id, group_id in det_cluster.items():
                        params = json.dumps({"kf_id": kf_id, "group_id": group_id})                                                                                                                     
                        cursor.execute("""                                                                                                                                                              
                            SELECT * FROM cypher('maze', $$
                                MATCH (kf:Keyframe {keyframe_id: $kf_id})-[:HAS_POSE]->(p:Pose)                                                                                                         
                                SET p.dbscan_group = $group_id                                                                                                                                          
                                RETURN p
                            $$, %s::agtype) AS (p agtype)                                                                                                                                               
                        """, (params,))

                    conn.commit()    
                    print(f"DBSCAN complete. Assigned {len(det_cluster)} keyframes.", flush=True)
        
        except Exception as e:
          print(f"DBSCAN thread exception: {e}", flush=True)                                                                                                                                  
          import traceback
          traceback.print_exc()


    # End of dbscan().

def slam_counter(sample, n=10, count=[0]):
    # Fire dbscan every n slam topic receipts.
   
    # Increment counter.
    count[0] += 1

    # Fire every n times.
    if count[0] % n == 0:
        threading.Thread(target=dbscan, daemon=True).start()

    
# End of slam_counter().

if __name__ == "__main__":
    
    # Instantiate bot.
    turtlebot = Turtlebot(table, show_camera=False)

    # Run dbscan.

    # Subscribe to Zenoh topic. 
    sub = turtlebot.session.declare_subscriber(                                                                                                                                             
          "tb/slam/pose",
          slam_counter
      )     
    
    
    detect_objects(turtlebot)

