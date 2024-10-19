#!/bin/bash

# For debugging, use --progress=plain option
cp ../requirements.txt .
docker build . -t vit:latest
rm requirements.txt
