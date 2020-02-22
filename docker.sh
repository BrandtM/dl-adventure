#!/usr/bin/env bash

cd docker/python/dcgan
docker image build -t python-dcgan:latest .

cd trainer
docker image build -t python-dcgan-trainer:latest .

cd ../generator
docker image build -t python-dcgan-generator:latest .

cd ../../../..

# For now uncomment these lines to enable training/generating
# docker run --gpus all --rm -it -v `pwd`/python/dcgan:/application python-dcgan-trainer:latest
#docker run --rm -it -v `pwd`/python/dcgan:/application python-dcgan-generator:latest