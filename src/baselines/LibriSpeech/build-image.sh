#!/bin/bash

docker rmi zhuangweikang/librispeech-baseline:latest
docker build -t zhuangweikang/librispeech-baseline:latest .
docker push zhuangweikang/librispeech-baseline:latest