#!/bin/bash

# For debugging, use --progress=plain option
DOCKER_BUILDKIT=1 docker build . -t frozenbilm:latest
