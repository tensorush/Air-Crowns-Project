FROM ubuntu:18.04 

# Timezone config
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Package update and prerequisites
RUN apt-get update \
    && apt-get --assume-yes install software-properties-common \
    && apt-get --assume-yes install wget \
    && apt-get install -y git \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get --assume-yes install python3.9 \
    && apt-get --assume-yes install python3.9-distutils \
    && apt-get --assume-yes install libpython3.9-dev \
    && apt-get --assume-yes install python3-pip

WORKDIR /ml_model
COPY . .

# Upgrade pip
RUN python3.9 -m pip install --upgrade pip \
    && python3.9 -m pip install --upgrade distlib \
    && python3.9 -m pip install --upgrade setuptools

RUN python3.9 -m pip install torch==1.9.1 \
    && python3.9 -m pip install torchvision==0.10.0 \
    && python3.9 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
