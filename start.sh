#!/bin/bash                                                                                                                                                                                 
  set -e                                                                                                                                                                                      
                                                                                                                                                                                              
  echo "Starting zenoh-router..."                                                                                                                                                             
  docker compose up -d zenoh-router                                                                                                                                                           
  sleep 3                                                                                                                                                                                     
   
  echo "Starting demo-world-enhanced..."                                                                                                                                                      
  docker compose up -d demo-world-enhanced
  sleep 10

  echo "Starting zenoh-bridge..."
  docker compose up -d zenoh-bridge
  sleep 5

  echo "Starting demo-slam..."
  docker compose up -d demo-slam
  sleep 3                                                                                                                                                                                     
   
  echo "Starting slam-logger..."                                                                                                                                                              
  docker compose up -d slam-logger

  echo "All containers up."
