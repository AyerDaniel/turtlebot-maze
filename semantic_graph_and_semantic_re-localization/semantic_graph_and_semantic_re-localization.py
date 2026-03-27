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
import hashlib

from psycopg.rows import dict_row
import clip
from PIL import Image as PILImage
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
            "event_id": str(uuid.uuid4()),  # unique event ID
            "run_id": str(uuid.uuid4()),    # unique run ID
            "robot_id": "tb3_sim",
            "sequence": int(),
            "image": {},
            "odometry": {},
            "tf": {},
            "detections": {},
            "pose": {},
            "slam": {},
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
        self.sub_maze_detections = self.session.declare_subscriber("maze/**/detections/v1/*", self.write_detections)
        print("Subscribed to maze/**/detections/v1/*", flush=True)

        # Subscribe to odom.
        self.sub_robot_state = self.session.declare_subscriber("odom", self.odom_callback)
        print(f"Subscribed to: odom")

        # Subscribe to TF
        self.sub_TF = self.session.declare_subscriber("tf_static", self.tf_callback)
        print(f"Subscribed to: tf")

        # Subscribe to SLAM.
        self.sub_slam = self.session.declare_subscriber("tb/slam/pose", self.slam_pose_callback)
        print(f"Subscribed to slam/pose")

        self.sub_slam = self.session.declare_subscriber("tb/slam/status", self.slam_status_callback)
        print(f"Subscribed to slam/status")
                                                                                                                                                     
    def rotation_delta_rad(self, mat_prev, mat_curr):  

        # Extract 3x3 matrix of rotational values from SLAM Pose Zenoh Topic.                                                                                                                                               
        R_prev = mat_prev[:3, :3]                                                                                                                                                               
        R_curr = mat_curr[:3, :3]                                

        # Compute rotation between matrices.  For rotational matrix transpose is inverse as per documentation.                                                                                                                               
        R_rel = R_curr @ R_prev.T

        # Get cos of angle.
        cos_angle = np.clip((np.trace(R_rel) - 1) / 2, -1, 1)                                                                                                                                   
        
        return np.arccos(cos_angle)        

    def slam_pose_callback(self, sample):

        # Get JSON from Zenoh Topic.
        data = json.loads(bytes(sample.payload))

        # Check timestamps to skip or not.
        try:
            
            # Rate limit check - if less than 100 ms since the last frame was considered, skip immediately #

            # Set time window to ignore new images in ms.
            del_t = 100  # In miliseconds.

            # Treat self.data['slam']['timestamp'] as current time keeper. 0.01 coefficient converts del_t into ms.
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

            # Run YOLO inference on a single image.
            results = self.yolo_model(curr_image)

            for result in results:
                
                # See if there is a detection.
                if len(result.boxes) == 0:
                    # Nothing detected.
                    continue
                    # Should we embed this anyway for later localization?
                
                else:

                    # Get box coords of detected object.
                    for box in result.boxes.xyxy:                                                                                                                                                               
                        x1, y1, x2, y2 = box.tolist() 
                    
                        # Get detected classes.
                        classes = box.cls.tolist()

                        # Get confidence.
                        confidence = box.conf.tolist()

                        # Get names.
                        class_names = result.names  

                        # Read in image.  Expected encoding: 'rgb8, bytes: 19437' output from ROS2.
                        nparr = np.frombuffer(curr_image, np.uint8)                                                                                                                                                  
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)   

                        # Crop image.
                        cropped = img[int(y1):int(y2), int(x1):int(x2)]

                        '''
                            The assignment says to:
                            Batch all crops through the CLIP encoder (ViT-B/32) to produce 512-dim L2-normalized embeddings

                            However, I am going to create the embeds as the image comes.  The business logic to batch the clips 
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


                        '''
                            Publish a JSON envelope to tb/detections containing:
                                keyframe_id - monotonically increasing integer
                                timestamp - wall clock time
                                map_x, map_y, map_yaw - robot pose in map frame
                                detections - array of \\{class, confidence, bbox, embedding, embedding_dim, embedding_model\\}

                        '''

                        # Create JSON envelope.
                        json_envelope = {
                            
                                "keyframe_id": str(uuid.uuid4()),  # unique event ID
                                "timestamp": datetime.now().isoformat(),
                                "map_x": self.data["odometry"]["x"],
                                "map_y": self.data["odometry"]["y"],
                                "map_yaw": self.data["odometry"]["yaw"],
                                "detections": [{
                                    'class': classes,
                                    'confidence': confidence,
                                    'bbox': (x1, y1, x2, y2),
                                    'embedding': image_features.tolist(),
                                    'embedding_dims':512,
                                    'embedding_model': 'CLIP: ViT-B/32'

                                }]
                                
                        }

                        # TEsting.
                        for thing in json_envelope:
                            #print(type(thing))
                            print("thing")

                        # Set topic.
                        topic = "tb/detections"

                        # Serialize the data to JSON
                        serialized_data = json.dumps(json_envelope)

                        # Publish topic to Zenoh.
                        self.session.put(topic, serialized_data.encode())

        except (KeyError, TypeError):
            # Timestamp not set.
            try:

                # Set timestamp.
                self.data['pose']['timestamp'] = self.data['slam']['timestamp']

            except Exception as e:

                # Report error.
                print(f"Exception thrown: {e}")           

    def slam_status_callback(self, sample):

        # Process JSON.
        data = json.loads(bytes(sample.payload))

        # Get timestamp from SLAM status.
        self.data['slam']['timestamp'] = data['timestamp']
        
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
        self.data["event_id"] = str(uuid.uuid4())
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

def detect_objects(turtlebot):

    '''
        
        This function assumes zenoh-router, zenoh-bridge, dectector, demo-world-enhanced containers are running.

    '''

    # Init rclpy.
    rclpy.init() 

    # Run pipeline.
    turtlebot.run()

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
                pil_image = PILImage.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

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
    
    # Instantiate bot.
    turtlebot = Turtlebot(table, show_camera=False)

    # Run steps in assignments.
    # detect_objects(turtlebot)
    # create_embeddings()
    # do_age()

