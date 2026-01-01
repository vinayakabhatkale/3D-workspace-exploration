#!/bin/bash

IMAGE_NAME=3d-workspace-explorer
CONTAINER_NAME=workspace_container

# Stop and remove old container if exists
docker stop $CONTAINER_NAME 2>/dev/null
docker rm $CONTAINER_NAME 2>/dev/null

# Build image
docker build -t $IMAGE_NAME .

# Run new container with interactive login shell
docker run -it --name $CONTAINER_NAME $IMAGE_NAME bash -l

