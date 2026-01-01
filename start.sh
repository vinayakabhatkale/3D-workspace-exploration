#!/bin/bash
CONTAINER_NAME=workspace_container

# Check if the container exists
if [ "$(docker ps -a -q -f name=^/${CONTAINER_NAME}$)" ]; then
  # If it’s not running, start it (no attach yet)
  if [ -z "$(docker ps -q -f name=^/${CONTAINER_NAME}$)" ]; then
    echo "Starting container…"
    docker start $CONTAINER_NAME >/dev/null
  fi

  # Now exec into the running container
  echo "Entering container…"
  docker exec -it $CONTAINER_NAME bash -l
else
  echo "Container does not exist. Use ./restart.sh to build and run it."
  exit 1
fi

