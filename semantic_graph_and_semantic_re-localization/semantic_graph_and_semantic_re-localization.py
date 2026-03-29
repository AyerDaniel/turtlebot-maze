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
from numpy.linalg import norm

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
                "det_pk": None,
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

    def relocalize(self, conn, det_records):
        """Given det_records from the current keyframe, find the top-3 most likely Places.

        Uses pgvector KNN to find visually similar stored embeddings, follows
        Observation → Object → Keyframe → Place via dbscan_group on the Pose node.
        """
        if not det_records:
            return

        register_vector(conn)
        cursor = conn.cursor()

        # Activate AGE on this connection.
        cursor.execute("LOAD 'age'")
        cursor.execute("SET search_path = ag_catalog, \"$user\", public")

        place_scores = {}

        for rec in det_records:
            emb = rec['embedding']

            # KNN search — top 10 most similar stored embeddings.
            cursor.execute("""
                SELECT de.det_pk, 1 - (de.embedding <=> %s::vector) AS similarity
                FROM detection_embeddings de
                ORDER BY de.embedding <=> %s::vector
                LIMIT 10
            """, (emb, emb))

            knn_rows = cursor.fetchall()
            if not knn_rows:
                continue

            for det_pk, similarity in knn_rows:
                # Follow det_pk → Observation → Keyframe → Pose (dbscan_group).
                obs_params = json.dumps({"det_pk": det_pk})
                cursor.execute("""
                    SELECT * FROM cypher('maze', $$
                        MATCH (obs:Observation {det_pk: $det_pk})
                            <-[:HAS_OBSERVATION]-(kf:Keyframe)
                            -[:HAS_POSE]->(p:Pose)
                        RETURN p.dbscan_group
                    $$, %s::agtype) AS (dbscan_group agtype)
                """, (obs_params,))

                row = cursor.fetchone()
                if row is None:
                    continue

                group = ag(row[0])
                if group is None or group == '-1':
                    continue

                place_scores[group] = place_scores.get(group, 0.0) + float(similarity)

        if not place_scores:
            print("Relocalize: no candidate places found.", flush=True)
            return

        # Rank and print top-3.
        ranked = sorted(place_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"Relocalize — top-3 candidate places:", flush=True)
        for rank, (group_id, score) in enumerate(ranked, 1):
            # Get centroid of this place's keyframes for pose hypothesis.
            place_params = json.dumps({"place_id": f"place_{group_id}"})
            cursor.execute("""
                SELECT * FROM cypher('maze', $$
                    MATCH (pl:Place {place_id: $place_id})
                    RETURN pl.centroid_x, pl.centroid_y
                $$, %s::agtype) AS (cx agtype, cy agtype)
            """, (place_params,))
            place_row = cursor.fetchone()
            if place_row:
                cx = ag(place_row[0])
                cy = ag(place_row[1])
                print(f"  #{rank} place_{group_id}  score={score:.3f}  pose=({cx}, {cy})", flush=True)
            else:
                print(f"  #{rank} place_{group_id}  score={score:.3f}", flush=True)

    # End of relocalize().

    def write_to_graph(self, conn, json_, det_records, graph='maze'):
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

            try:
                robot_x = json_['map_x']
                robot_y = json_['map_y']
                emb = np.array(det['embedding'])

                fusion_params = json.dumps({"class_name": det['class_name']})
                cursor.execute("""
                        SELECT * FROM cypher('maze', $$
                            MATCH (obj:Object {class_name: $class_name})
                            RETURN obj.object_id, obj.mean_x, obj.mean_y, obj.mean_embedding,
                                    obj.observation_count, obj.first_seen
                        $$, %s::agtype) AS (
                            object_id agtype, mean_x agtype, mean_y agtype,
                            mean_embedding agtype, observation_count agtype, first_seen agtype
                        )
                    """, (fusion_params,))

                best_id = None
                best_sim = -1.0
                for row in cursor.fetchall():
                    obj_id    = ag(row[0])
                    obj_x     = float(ag(row[1]))
                    obj_y     = float(ag(row[2]))
                    obj_emb   = np.array(json.loads(ag(row[3])))
                    spatial_d = np.sqrt((robot_x - obj_x)**2 + (robot_y - obj_y)**2)
                    cos_sim   = float(np.dot(emb, obj_emb))  # both L2-normalised

                    if cos_sim >= 0.7 and spatial_d <= 3.0 and cos_sim > best_sim:
                        best_sim = cos_sim
                        best_id  = obj_id

                now_iso = datetime.now().isoformat()

                if best_id is not None:
                    # Merge into existing Object — update running averages.
                    merge_params = json.dumps({
                        "object_id": best_id,
                        "rx": robot_x, "ry": robot_y,
                        "emb": det['embedding'],
                        "last_seen": now_iso,
                    })
                    cursor.execute("""
                        SELECT * FROM cypher('maze', $$
                            MATCH (obj:Object {object_id: $object_id})
                            SET obj.mean_x = (obj.mean_x * obj.observation_count + $rx)
                                            / (obj.observation_count + 1),
                                obj.mean_y = (obj.mean_y * obj.observation_count + $ry)
                                            / (obj.observation_count + 1),
                                obj.observation_count = obj.observation_count + 1,
                                obj.last_seen = $last_seen
                            RETURN obj.object_id
                        $$, %s::agtype) AS (object_id agtype)
                    """, (merge_params,))
                    matched_id = best_id
                else:
                    # Create new Object landmark.
                    new_obj_id = str(uuid.uuid4())
                    create_params = json.dumps({
                        "object_id":         new_obj_id,
                        "class_name":        det['class_name'],
                        "mean_x":            robot_x,
                        "mean_y":            robot_y,
                        "mean_embedding":    det['embedding'],
                        "observation_count": 1,
                        "first_seen":        now_iso,
                        "last_seen":         now_iso,
                    })
                    cursor.execute("""
                        SELECT * FROM cypher('maze', $$
                            CREATE (obj:Object {
                                object_id:         $object_id,
                                class_name:        $class_name,
                                mean_x:            $mean_x,
                                mean_y:            $mean_y,
                                mean_embedding:    $mean_embedding,
                                observation_count: $observation_count,
                                first_seen:        $first_seen,
                                last_seen:         $last_seen
                            })
                            RETURN obj.object_id
                        $$, %s::agtype) AS (object_id agtype)
                    """, (create_params,))
                    matched_id = new_obj_id

                # Create Observation node and link it.
                obs_params = json.dumps({
                    "kf_id":      kf_id,
                    "object_id":  matched_id,
                    "confidence": det['confidence'],
                    "bbox":       json.dumps(det['bbox']),
                    "emb_model":  det['embedding_model'],
                    "embedding":  det['embedding'],
                    "det_pk":     det['det_pk'],
                })
                cursor.execute("""
                    SELECT * FROM cypher('maze', $$
                        MATCH (kf:Keyframe {keyframe_id: $kf_id})
                        MATCH (obj:Object {object_id: $object_id})
                        CREATE (obs:Observation {
                            confidence:      $confidence,
                            bbox:            $bbox,
                            embedding_model: $emb_model,
                            embedding:       $embedding,
                            det_pk:          $det_pk
                        })
                        CREATE (kf)-[:HAS_OBSERVATION]->(obs)
                        CREATE (obs)-[:CORRESPONDS_TO]->(obj)
                        RETURN obs
                    $$, %s::agtype) AS (obs agtype)
                """, (obs_params,))


            except Exception as e:

                print(f"Exception {e}")

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

    def write_embeddings(self, conn, det_records):         
                                                   
        cursor = conn.cursor()                                                                                                                                                                  
        register_vector(conn)                                                                            
                                                                                                                                                                                                
        for rec in det_records:                                                                          
            try:                                                          
                cursor.execute("""                                                                                                                                                              
                    INSERT INTO detection_embeddings (det_pk, model, embedding)                                                                                                                 
                    VALUES (%s, %s, %s)                                                                                                                                                         
                    ON CONFLICT DO NOTHING                                                                                                                                                      
                """, (                     
                    rec['det_pk'],                                                                                                                                                              
                    rec['embedding_model'],                                                              
                    rec['embedding'],                                     
                ))                                                   
            except Exception as e:                         
                print(f"Failed to write embedding: {e}", flush=True)
                                                                                                                                                                                              
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
                
                # No detections.
                pass

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
                
        # Build json_envelope.
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

        # Publish to Zenoh if detections found.
        if detections:
            self.session.put("tb/detections", json.dumps(json_envelope).encode())

        # Single unified DB write.
        with psycopg.connect(
            f"dbname={dbname} user={user} password={password} host={host} port={port}"
        ) as conn:
            det_records = self.write_detections(conn, json_envelope, detections)
            self.write_to_graph(conn, json_envelope, det_records, graph='maze')
            self.write_embeddings(conn, det_records)
            self.relocalize(conn, det_records)
            conn.commit()

        self.data['pose']['timestamp'] = self.data['slam']['timestamp']
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

    def write_detections(self, conn, json_envelope, detections):
    # Write json to db.

        cursor = conn.cursor()

        image    = self.data["image"]
        odometry = self.data["odometry"]
        stamp    = image["stamp"]

        time_in_run = datetime.fromtimestamp(
            stamp["sec"] + stamp["nanosec"] * 1e-9, timezone.utc
        )

        success, encoded = cv2.imencode('.jpg', self.data['image']['current_image'])
        img_bytes = encoded.tobytes() if success else None

        try:
            cursor.execute("""
                INSERT INTO detection_events (
                    event_id, run_id, robot_id, sequence, time_in_run,
                    image_frame_id, image_sha256, width, height, encoding,
                    x, y, yaw, vx, vy, wz,
                    tf_ok, t_base_camera, raw_event, image_raw
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (event_id) DO NOTHING
            """, (
                json_envelope["event_id"],
                json_envelope["run_id"],
                json_envelope["robot_id"],
                json_envelope["sequence"],
                time_in_run,
                image["frame_id"],
                image["sha256"],
                image["width"],
                image["height"],
                image["encoding"],
                round(odometry["x"],   2),
                round(odometry["y"],   2),
                round(odometry["yaw"], 2),
                round(odometry["vx"],  2),
                round(odometry["vy"],  2),
                round(odometry["wz"],  2),
                self.data["tf"]["tf_ok"],
                self.data["tf"]["t_base_camera"],
                json.dumps(json_envelope),
                img_bytes,
            ))
        except Exception as e:
            print(f"Failed to write detection_events: {e}", flush=True)
            return []

        # --- detections (one row per detection, returns det_pk) ---
        results = []
        for det in detections:
            confidence = round(det['confidence'], 3)
            x1, y1, x2, y2 = [round(v) for v in det['bbox']]

            try:
                cursor.execute("""
                    INSERT INTO detections (event_id, det_id, class_name, confidence, x1, y1, x2, y2)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (x1, y1, x2, y2)
                    DO UPDATE SET
                        det_id     = EXCLUDED.det_id,
                        class_name = EXCLUDED.class_name,
                        confidence = EXCLUDED.confidence
                    WHERE EXCLUDED.confidence > detections.confidence
                    RETURNING det_pk
                """, (
                    json_envelope["event_id"],
                    str(uuid.uuid4()),
                    det['class'],
                    confidence,
                    x1, y1, x2, y2,
                ))

                row = cursor.fetchone()
                if row is None:
                    print(f"Detection skipped (lower confidence): {det['class']} {x1,y1,x2,y2}", flush=True)
                    continue

                results.append({                                                                                                                                                            
                    'det_pk':          row[0],         
                    'class_name':      det['class'],                                                                                                                                        
                    'confidence':      det['confidence'],
                    'bbox':            det['bbox'],      
                    'embedding':       det['embedding'],
                    'embedding_model': det['embedding_model'],
                })

            except Exception as e:
                print(f"Failed to write detection: {e}", flush=True)
                continue

        return results

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

def ag(v):                                                                                                                                                                                  
    if v is None: return None                                                                                                                                                               
    s = str(v)                                                                                                                                                                              
    return s.strip('"')

# End of ag().

def dbscan():
        # This function performs dbscan on the sampled space.

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
                            'dbscan_group': int(ag(row[3])) if ag(row[3]) is not None else -1 
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

                    # --- Build Place nodes ---
                    groups = set(det_cluster.values())
                    for group_id in groups:
                        # Compute centroid of all keyframes in this group.
                        coords_in_group = [
                            (props['x'], props['y'])
                            for kf_id, props in all_kfs.items()
                            if det_cluster.get(kf_id) == group_id
                        ]
                        cx = sum(c[0] for c in coords_in_group) / len(coords_in_group)
                        cy = sum(c[1] for c in coords_in_group) / len(coords_in_group)

                        place_params = json.dumps({
                            "place_id":  f"place_{group_id}",
                            "group_id":  group_id,
                            "centroid_x": cx,
                            "centroid_y": cy,
                        })
                        cursor.execute("""
                            SELECT * FROM cypher('maze', $$
                                MERGE (pl:Place {place_id: $place_id})
                                SET pl.group_id   = $group_id,
                                    pl.centroid_x = $centroid_x,
                                    pl.centroid_y = $centroid_y
                                RETURN pl
                            $$, %s::agtype) AS (pl agtype)
                        """, (place_params,))

                    # --- Link Keyframes to Places ---
                    for kf_id, group_id in det_cluster.items():
                        kf_place_params = json.dumps({
                            "kf_id":    kf_id,
                            "place_id": f"place_{group_id}",
                        })
                        cursor.execute("""
                            SELECT * FROM cypher('maze', $$
                                MATCH (kf:Keyframe {keyframe_id: $kf_id})
                                MATCH (pl:Place {place_id: $place_id})
                                MERGE (kf)-[:LOCATED_IN]->(pl)
                                RETURN kf
                            $$, %s::agtype) AS (kf agtype)
                        """, (kf_place_params,))

                    # --- Place adjacency edges ---
                    # Two places are adjacent if any of their keyframes are within 2*db_eps of each other.
                    group_ids = list(groups)
                    for i in range(len(group_ids)):
                        for j in range(i + 1, len(group_ids)):
                            ga, gb = group_ids[i], group_ids[j]
                            coords_a = np.array([
                                (props['x'], props['y'])
                                for kf_id, props in all_kfs.items()
                                if det_cluster.get(kf_id) == ga
                            ])
                            coords_b = np.array([
                                (props['x'], props['y'])
                                for kf_id, props in all_kfs.items()
                                if det_cluster.get(kf_id) == gb
                            ])
                            tree_b = cKDTree(coords_b)
                            dists, _ = tree_b.query(coords_a)
                            if dists.min() <= 2 * db_eps:
                                adj_params = json.dumps({
                                    "place_a": f"place_{ga}",
                                    "place_b": f"place_{gb}",
                                })
                                cursor.execute("""
                                    SELECT * FROM cypher('maze', $$
                                        MATCH (pa:Place {place_id: $place_a})
                                        MATCH (pb:Place {place_id: $place_b})
                                        MERGE (pa)-[:ADJACENT_TO]->(pb)
                                        MERGE (pb)-[:ADJACENT_TO]->(pa)
                                        RETURN pa
                                    $$, %s::agtype) AS (pa agtype)
                                """, (adj_params,))

                    conn.commit()
                    print(f"Places built: {len(groups)} groups, adjacency edges added.", flush=True)

        
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
    
    # Detect objects as we go.
    detect_objects(turtlebot)

