#!/bin/bash

if [ $1 == "manager" ]
then
    docker rmi -f dlcpod-dev:manager
    docker rmi -f centaurusinfra/dlcpod-dev:manager
    docker build -t dlcpod-dev:manager -f manager/Dockerfile .
    docker tag dlcpod-dev:manager centaurusinfra/dlcpod-dev:manager
    docker push centaurusinfra/dlcpod-dev:manager
else
    docker rmi -f dlcpod-dev:client
    docker rmi -f centaurusinfra/dlcpod-dev:client
    docker build -t dlcpod-dev:client -f client/Dockerfile .
    docker tag dlcpod-dev:client centaurusinfra/dlcpod-dev:client
    docker push centaurusinfra/dlcpod-dev:client  
fi