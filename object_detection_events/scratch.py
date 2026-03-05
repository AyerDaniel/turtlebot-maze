# Detection callback pulls latest image from queue
    def detections_callback(self, sample):

        if self.image_queue.empty():

            #print("No image in queue yet", flush=True)
            return

        try:

            detections = json.loads(sample.payload.to_bytes())

        except Exception as e:

            print(f"Failed to parse detection: {e}", flush=True)
            return

        if not detections:
            return

        detection = detections[0]

        # Get the latest image from the queue
        try:
            img = self.image_queue.get_nowait()

        except:

            print("Failed to get image from queue", flush=True)
            return

        # Build parameterized SQL
        columns = []
        values = []
        placeholders = []

        for key in detection:
            match key:
                case "class":
                    columns.append("class")
                    values.append(detection[key])
                    placeholders.append("%s")
                case "confidence":
                    columns.append("confidence")
                    values.append(detection[key])
                    placeholders.append("%s")
                case "bbox":
                    columns.append("bbox")
                    values.append(detection[key])
                    placeholders.append("%s")

        # Store image as PNG bytes
        columns.append("image")
        _, buffer = cv2.imencode(".png", img)
        values.append(buffer.tobytes())
        placeholders.append("%s")

        write_row_sql = f"INSERT INTO {self.table} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"

        try:
            with psycopg.connect(
                f"dbname={dbname} user={user} password={password} host={host} port={port}"
            ) as conn:
                with conn.cursor() as cur:
                    cur.execute(write_row_sql, values)
                    conn.commit()
            print(f"Inserted detection for class '{detection.get('class')}'", flush=True)
        except Exception as e:
            print(f"DB insert error: {e}", flush=True)