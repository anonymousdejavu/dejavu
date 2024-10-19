#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
IMAGE=vit
TAG=latest
NAME=vit

docker run -it \
    --net=host \
    -v ${SCRIPT_DIR}/..:/workspace \
    -v /mnt/nfs:/mnt/nfs \
    -v /mnt/ssd1:/mnt/ssd1 \
    -v /mnt/ssd2:/mnt/ssd2 \
    -v /mnt/ssd3:/mnt/ssd3 \
    -v /mnt/ssd4:/mnt/ssd4 \
    -v /mnt/hdd1:/mnt/hdd1 \
    -v /mnt/hdd2:/mnt/hdd2 \
    --gpus all \
    --ulimit core=-1 --privileged \
    --security-opt seccomp=unconfined \
    --cap-add=SYS_PTRACE \
    --shm-size=110gb \
    --name=${NAME} \
    ${IMAGE}:${TAG} bash
