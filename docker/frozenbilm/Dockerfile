# syntax = docker/dockerfile:1.0-experimental
FROM continuumio/conda-ci-linux-64-python3.8:latest AS base-builder

USER root


WORKDIR /root
ADD requirements.txt requirements.txt
RUN conda install pytorch==1.8.1 torchvision==0.9.1 cudatoolkit=11.1 -c pytorch -c nvidia
RUN python -m pip install -r requirements.txt
RUN python -m pip install hostlist
RUN python -m pip install safetensors transformers==4.26.1


ENV LC_ALL=C.UTF-8
WORKDIR /workspace

ENTRYPOINT [ "zsh", "-l" ]

