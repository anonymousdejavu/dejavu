#!/bin/bash
cp ../../requirements.txt requirements.txt
docker build . -t diffrate
rm requirements.txt
