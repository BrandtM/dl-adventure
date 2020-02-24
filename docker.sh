#!/usr/bin/env bash

cd docker/python/dcgan
docker image build -t python-dcgan:latest .

cd trainer
docker image build -t python-dcgan-trainer:latest .

cd ../../../..

docker run --gpus all --rm -it -v `pwd`/python/dcgan:/application python-dcgan-trainer:latest