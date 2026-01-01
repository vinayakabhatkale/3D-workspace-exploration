#!/bin/bash
IMAGE_NAME=3d-workspace-explorer

echo "Building Docker image: $IMAGE_NAME ..."
docker build -t $IMAGE_NAME .

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
else
    echo "❌ Build failed!"
fi
