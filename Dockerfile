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
    && apt-get --assume-yes install python3-pip \
    && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /it_crowns_solution
COPY . .

# Upgrade pip
RUN python3.9 -m pip install --upgrade pip \
    && python3.9 -m pip install --upgrade distlib \
    && python3.9 -m pip install --upgrade setuptools

RUN python3.9 -m pip install Flask \
    && python3.9 -m pip install numpy \
    && python3.9 -m pip install opencv-python \
    && python3.9 -m pip install pandas \
    && python3.9 -m pip install --upgrade tensorflow \
    && python3.9 -m pip install keras \
    && python3.9 -m pip install Pillow
WORKDIR /it_crowns_solution/web

CMD python3.9 app.py
