#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
IMAGE=diffrate
TAG=latest
NAME=diffrate
NFS_DIR=/mnt

docker run -it \
    --net=host \
    -v ${SCRIPT_DIR}/../..:/workspace \
    -v ${NFS_DIR}/ssd1:/mnt/ssd1 \
    -v ${NFS_DIR}/ssd2:/mnt/ssd2 \
    -v ${NFS_DIR}/ssd3:/mnt/ssd3 \
    -v ${NFS_DIR}/ssd4:/mnt/ssd4 \
    -v ${NFS_DIR}/hdd1:/mnt/hdd1 \
    -v ${NFS_DIR}/hdd2:/mnt/hdd2 \
    -v ${NFS_DIR}/nfs:/mnt/nfs \
    --gpus all \
    --ulimit core=-1 --privileged \
    --security-opt seccomp=unconfined \
    --cap-add=SYS_PTRACE \
    --shm-size=110gb \
    --name=${NAME} \
    ${IMAGE}:${TAG} bash
