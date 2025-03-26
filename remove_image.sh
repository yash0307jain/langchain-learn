#!/bin/bash

# The image name or ID provided as argument
IMAGE_NAME_OR_ID="langchain-guide-chatbot"

# Stop all containers using the specified image
echo "Stopping containers using image: $IMAGE_NAME_OR_ID..."
docker ps -q --filter "ancestor=$IMAGE_NAME_OR_ID" | xargs -r docker stop

# Remove all containers using the specified image
echo "Removing containers using image: $IMAGE_NAME_OR_ID..."
docker ps -a -q --filter "ancestor=$IMAGE_NAME_OR_ID" | xargs -r docker rm

# Remove the specific image
echo "Removing image: $IMAGE_NAME_OR_ID..."
docker rmi $IMAGE_NAME_OR_ID

# # Clean up unused Docker resources (volumes, networks, etc.)
# echo "Cleaning up unused Docker resources..."
# docker system prune -af

# # Optional: Clean up dangling images
# echo "Cleaning up dangling images..."
# docker image prune -f

# echo "Docker cleanup for image $IMAGE_NAME_OR_ID complete."
