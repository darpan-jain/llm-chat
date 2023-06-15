#!/bin/sh

docker run -i --rm -dP -v <insert source path>:/app/chat.log --name $1 chat_demo:latest && docker logs -f $1