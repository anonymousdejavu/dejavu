FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

USER root
WORKDIR /root

# RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

RUN python -m pip install -U pip

# Install PyTorch 2.0
RUN python -m pip install --default-timeout=1800 \
    termcolor \
    timm==0.8.0.dev0 \
    torchvision \
    scipy

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN python -m pip install \
    datasets \
    transformers \
    opencv-python \
    joblib

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install -r /tmp/requirements.txt

RUN apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6

##########################
#  Setup personal tools
##########################

RUN apt update \
    && apt install -y \
        git \
        curl \
        wget


RUN git config --global --add safe.directory /workspace

ENV LC_ALL=C.UTF-8
ENV LD_LIBRARY_PATH="/opt/tritonserver/lib:${LD_LIBRARY_PATH}:/usr/lib/x86_64-linux-gnu"
ENV PATH="/usr/local/bin:${PATH}"
WORKDIR /workspace
