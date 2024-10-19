#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
IMAGE=frozenbilm
TAG=latest
NAME=frozenbilm
NFS_DIR=/mnt/nfs

docker run -it \
    --gpus all \
    --net=host \
    -v ${SCRIPT_DIR}/../..:/workspace \
    -v ${NFS_DIR}/ssd1:/mnt/ssd1 \
    -v ${NFS_DIR}/ssd2:/mnt/ssd2 \
    -v ${NFS_DIR}/ssd3:/mnt/ssd3 \
    -v ${NFS_DIR}/ssd4:/mnt/ssd4 \
    -v ${NFS_DIR}/hdd1:/mnt/hdd1 \
    -v ${NFS_DIR}/hdd2:/mnt/hdd2 \
    --shm-size=110gb \
    --name=${NAME} \
    ${IMAGE}:${TAG}
