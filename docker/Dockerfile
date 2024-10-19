# syntax = docker/dockerfile:1.0-experimental
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

USER root

RUN python -m pip install transformers accelerate

RUN python -m pip install \
    joblib \
    peft \
    pandas

RUN python -m pip install \
    wandb \
    python-box

RUN python -m pip install \
    jupyterlab

RUN apt update && apt install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tmux

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install -r /tmp/requirements.txt

# Ugly, but only way found to install ffmpeg with libx264
RUN apt install -y ffmpeg \
    && rm /opt/conda/bin/ffmpeg

RUN apt install -y gnuplot

RUN git config --global --add safe.directory /workspace

ENV LC_ALL=C.UTF-8
WORKDIR /workspace