#!/bin/bash

# Head into working directory.
cd /home/johnsmith/Desktop/njit/workspaces/turtlebot-maze

# Stop containers. 
docker compose down zenoh-router
docker compose down zenoh-bridge
docker compose down demo-world-enhanced
docker compose down detector

# Move to project folder.
cd /home/johnsmith/Desktop/njit/workspaces/turtlebot-maze/object_detection_events

# Run containers.
docker compose down postgres
docker compose down object_detection_events

## All containers for the proejt should now be stopped.