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

WORKDIR /ml_model/tmp

# Installing pytorch-1.7.1 && torchvision-0.8.2
RUN wget \
    https://files.pythonhosted.org/packages/41/f4/4da4f26a04d93851e481e76ec17fed0d152a1691e8f1142ad763c9f07997/torch-1.7.1-cp39-cp39-manylinux1_x86_64.whl \
    && python3.9 -m pip install -q torch-1.7.1-cp39-cp39-manylinux1_x86_64.whl \
    && python3.9 -m pip install torchvision==0.8.2

# Installing detectron2
RUN git clone https://github.com/facebookresearch/detectron2.git \
    && python3.9 -m pip install detectron2

WORKDIR /ml_model

RUN rm -rf tmp
