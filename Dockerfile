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

RUN apt update && apt install -y ffmpeg



ENV LC_ALL=C.UTF-8
WORKDIR /workspace

ENTRYPOINT ["accelerate", "launch", \
    "--multi_gpu", \
    "--num_processes", "4", \
    "--mixed_precision", "bf16", \
    "--dynamo_backend", "eager", \
    "-m", "dejavu.train", "config/msrvtt/base.yaml"]

