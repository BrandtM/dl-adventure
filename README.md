# This is my adventure into deep learning  

Code won't work as-is. You'll have to source training data yourself.  
An image downloader has been provided for simplicity (it's not a crawler though).  

## Docker setup  

This docker setup is still in its early stages but for now it works.  
The directory `docker/python/dcgan` contains the Dockerfiles for the DCGAN. The top-level Dockerfile is the base GPU
image for the DCGAN. The `trainer/` directory contains the Dockerfile for the DCGAN trainer and the `generator/` directory
contains the Dockerfile for the generator. I wanted the generator to be lightweight and portable so it uses the CPU-only
variant of Tensorflow.  
See `docker.sh` for runtime instructions and image naming conventions.  

### Requirements  

This has only been tested on Arch Linux!  
* Docker >= 19.03
* NVIDIA and CUDA drivers
* [NVIDIA Container Toolkit](https://aur.archlinux.org/packages/nvidia-container-toolkit/)  

Restart Docker after installing the "Container Toolkit" or your GPU won't be recognized by Docker.

## Cats as a service

Work towards making my DCGAN a CaaS (Cats-as-a-Service) is ongoing but in its early stages.
Run `docker-compose up` to start a small gunicorn server behind a nginx reverse proxy,
then visit `localhost:8080`.

## Some outputs  

My ~50 images of cats resulted in this after ~280000 epochs:  

![Example image](https://raw.githubusercontent.com/BrandtM/dl-adventure/master/images/cats.png)  