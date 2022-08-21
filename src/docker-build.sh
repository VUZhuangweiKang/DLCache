#!/bin/bash

if [ $1 == "manager" ]
then
    docker rmi -f dlcpod-dev:manager
    docker rmi -f zhuangweikang/dlcpod-dev:manager
    docker build -t dlcpod-dev:manager -f manager/Dockerfile .
    docker tag dlcpod-dev:manager zhuangweikang/dlcpod-dev:manager
    docker push zhuangweikang/dlcpod-dev:manager
else
    docker rmi -f dlcpod-dev:client
    docker rmi -f zhuangweikang/dlcpod-dev:client
    docker build -t dlcpod-dev:client -f client/Dockerfile .
    docker tag dlcpod-dev:client zhuangweikang/dlcpod-dev:client
    docker push zhuangweikang/dlcpod-dev:client  
fi