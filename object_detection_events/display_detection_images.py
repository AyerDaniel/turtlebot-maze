#!/usr/bin/env python3
"""
Retrieve images and detections from PostgreSQL, overlay bounding boxes, and display using OpenCV.
"""

import psycopg
import cv2
import numpy as np
import json

# Import DB connection variables
from db_connect import *

def main():
    try:
        # Connect to the database
        with psycopg.connect(
            f"dbname={dbname} user={user} password={password} host={host} port={port}"
        ) as conn:

            # Prompt user for bbox.
            target_bbox = input(f"Please enter the coordinates for the bbox as four floats separated by commas.")

            img_sql = f"""SELECT image, class, confidence, bbox 
                    FROM detections 
                    WHERE bbox = ARRAY[{target_bbox}]::double precision[]"""
            
            # Testing
            print(img_sql)
            
            with conn.cursor() as cur:
                cur.execute(img_sql)
                rows = cur.fetchall()

                if not rows:
                    print("No images found in the database.")
                    return
                
                for i, (image_bytes, class_name, confidence, bbox) in enumerate(rows):
                    # Decode image from PNG bytes
                    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
                    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

                    if img is None:
                        print(f"Failed to decode image {i}", flush=True)
                        continue

                    # # Draw bounding box if available
                    # if bbox:
                    #     # bbox should be a list: [x_min, y_min, x_max, y_max]
                    #     try:
                    #         if isinstance(bbox, str):
                    #             # If stored as PostgreSQL array, convert string to list
                    #             bbox_list = json.loads(bbox.replace("'", '"'))
                    #         else:
                    #             bbox_list = list(bbox)

                    #         x_min, y_min, x_max, y_max = map(int, bbox_list)
                    #         cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    #         label = f"{class_name} ({confidence:.2f})"
                    #         cv2.putText(
                    #             img,
                    #             label,
                    #             (x_min, y_min - 10),
                    #             cv2.FONT_HERSHEY_SIMPLEX,
                    #             0.6,
                    #             (0, 255, 0),
                    #             2,
                    #         )
                    #     except Exception as e:
                    #         print(f"Failed to parse bbox for image {i}: {e}", flush=True)

                    # Display the image
                    cv2.imshow("Detection", img)
                    key = cv2.waitKey(0)  # Wait for key press to show next image
                    cv2.destroyAllWindows()

            # with conn.cursor() as cur:
            #     # Query last 50 entries (adjust as needed)
            #     cur.execute(f"SELECT image, class, confidence, bbox FROM {table} ORDER BY id DESC LIMIT 50")
            #     rows = cur.fetchall()

            #     if not rows:
            #         print("No images found in the database.")
            #         return

            #     # Display all images.
            #     for i, (image_bytes, class_name, confidence, bbox) in enumerate(rows):
            #         # Decode image from PNG bytes
            #         image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            #         img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            #         if img is None:
            #             print(f"Failed to decode image {i}", flush=True)
            #             continue

            #         # Draw bounding box if available
            #         if bbox:
            #             # bbox should be a list: [x_min, y_min, x_max, y_max]
            #             try:
            #                 if isinstance(bbox, str):
            #                     # If stored as PostgreSQL array, convert string to list
            #                     bbox_list = json.loads(bbox.replace("'", '"'))
            #                 else:
            #                     bbox_list = list(bbox)

            #                 x_min, y_min, x_max, y_max = map(int, bbox_list)
            #                 cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            #                 label = f"{class_name} ({confidence:.2f})"
            #                 cv2.putText(
            #                     img,
            #                     label,
            #                     (x_min, y_min - 10),
            #                     cv2.FONT_HERSHEY_SIMPLEX,
            #                     0.6,
            #                     (0, 255, 0),
            #                     2,
            #                 )
            #             except Exception as e:
            #                 print(f"Failed to parse bbox for image {i}: {e}", flush=True)

            #         # Display the image
            #         cv2.imshow("Detection", img)
            #         key = cv2.waitKey(0)  # Wait for key press to show next image
            #         cv2.destroyAllWindows()

    except Exception as e:
        print(f"Database error: {e}", flush=True)

if __name__ == "__main__":
    main()