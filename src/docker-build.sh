#!/bin/bash

if [ $1 == "manager" ]
then
    docker rmi -f dlcache-dev:manager
    docker rmi -f zhuangweikang/dlcache-dev:manager
    docker build -t dlcache-dev:manager -f manager/Dockerfile .
    docker tag dlcache-dev:manager zhuangweikang/dlcache-dev:manager
    docker push zhuangweikang/dlcache-dev:manager
elif [ $1 == "manager-worker" ] 
then
    docker rmi -f dlcache-dev:manager-worker
    docker rmi -f zhuangweikang/dlcache-dev:manager-worker
    docker build -t dlcache-dev:manager-worker -f manager-worker/Dockerfile .
    docker tag dlcache-dev:manager-worker zhuangweikang/dlcache-dev:manager-worker
    docker push zhuangweikang/dlcache-dev:manager-worker
else
    docker rmi -f dlcache-dev:client
    docker rmi -f zhuangweikang/dlcache-dev:client
    docker build -t dlcache-dev:client -f client/Dockerfile .
    docker tag dlcache-dev:client zhuangweikang/dlcache-dev:client
    docker push zhuangweikang/dlcache-dev:client  
fi