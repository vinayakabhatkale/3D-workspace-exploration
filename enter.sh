#!/bin/bash
CONTAINER_NAME=workspace_container

if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "Entering container..."
    docker exec -it $CONTAINER_NAME bash
else
    echo "Container not running. Start it using ./restart.sh"
fi
