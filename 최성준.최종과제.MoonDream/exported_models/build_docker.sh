#!/bin/bash
# Build and run Moondream Docker container

echo "Building Docker image..."
docker build -t moondream:latest .

echo "Running container..."
docker-compose up -d

echo "✅ Container is running!"
echo "Check logs: docker logs moondream-inference"
