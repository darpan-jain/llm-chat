#!/bin/sh

git fetch && git pull
docker build -t chat_demo:latest .
echo "Build complete for Chat Demo."