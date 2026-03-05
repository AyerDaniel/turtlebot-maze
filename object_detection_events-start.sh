#!/bin/bash

# Head into working directory.
cd /home/johnsmith/Desktop/njit/workspaces/turtlebot-maze

# Run containers.  -d to run headless.
docker compose up -d zenoh-router
docker compose up -d zenoh-bridge
docker compose up -d demo-world-enhanced
docker compose up -d detector

# Move to project folder.
cd /home/johnsmith/Desktop/njit/workspaces/turtlebot-maze/object_detection_events

# Run containers.
docker compose up -d postgres
docker compose up object_detection_events

##  You should now see detections made by the object_detection_events container in its terminal output.
##  You should now see entries made into the postgres database.