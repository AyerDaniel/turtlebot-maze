def do_age():

    '''

        This function assumes the detection_embeddings table has been populated appropriately.

    '''

    # Connect to db.
    try:
            
        with psycopg.connect(
                f"dbname={dbname} user={user} password={password} host={host} port={port}",
                row_factory=dict_row
            ) as conn:
            
            # Register type adapter to return embed vector as numpy array.
            register_vector(conn)

            # Create cursor.
            cursor = conn.cursor()

            # Load AGE.
            try:
                
                cursor.execute("LOAD 'age'")

            except Exception as e:

                print(f"Loading age threw exception: {e}")

            # Set search path.
            try:

                cursor.execute(f'SET search_path = ag_catalog, "$user", public')
            
            except Exception as e:
                
                print(f"Setting search path threw exception: {e}")

            # Get embeddings to analyze with DBSCAN.
            cursor.execute("SELECT det_pk, embedding FROM detection_embeddings")
            rows = cursor.fetchall()

            # Get embeddings with det_pks as identifiers.
            det_pks = [row['det_pk'] for row in rows]
            embeddings = np.array([row['embedding'] for row in rows])
            
            # Normalize embeddings.
            embeddings = normalize(embeddings)

            # Get count of possible calsses.
            cursor.execute("SELECT COUNT(DISTINCT class_name) FROM detections")                                                                                                                         
            target_clusters = cursor.fetchone()['count'] 
            
            # Report target clusters.
            print(f"Trying for {target_clusters} clusters.")

            # Run DBSCAN tuning eps to get the number of clusters to match the number of classes possible.
            for d_eps in range(1,10):
                    
                # Modify eps.
                eps = d_eps * 0.01 #<---- Try to modify the eps to find target clusters.

                # Run DBSCAN.  Set min_samples to 1 because all objects occur at least once.
                dbscan = DBSCAN(eps=eps, min_samples=1, metric='cosine')
                labels = dbscan.fit_predict(embeddings)

                # Number of clusters.
                num_clusters = len(set(labels)) - (1 if -1 in labels else 0) 

                # Count number of points labeled as noise.
                num_noise = list(labels).count(-1)                                                                                                                                                        
                                                                                                                                                                                              
                print(f"eps={eps:.2f}, clusters={num_clusters}, noise={num_noise}")                                                                                                                         
                                                                                                                                                                                                        
                if num_noise == 0:                                                                                                                                                                        
                    print(f"All points assigned at eps={eps:.2f}, {num_clusters} clusters.")
                    break         


            # Display results.
            for det_pk, label in zip(det_pks, labels):                                                                                                                                                  
                print(f"det_pk: {det_pk}, cluster: {label}")
    
        # Close cursor.
        cursor.close()

    except Exception as e:
        
        print(f"psycopg connection to db threw exception: {e}")