FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV LANG C.UTF-8
ARG FORCE_CUDA="1"
ENV FORCE_CUDA=${FORCE_CUDA}

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    software-properties-common \
    build-essential \
    openssh-server openssh-client\
    ffmpeg \
    g++ \
    htop \
    curl \
    locales \
    git \
    tar \
    python3-pip \
    python3-numpy \
    python3-scipy \
    nano \
    unzip \
    vim \
    wget \
    ca-certificates bzip2 cmake tree htop bmon iotop \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y \
            python3.9-venv \
            python3.9-dev 

ENV VIRTUAL_ENV=venv
RUN python3.9 -m venv /opt/$VIRTUAL_ENV
ENV PATH /opt/$VIRTUAL_ENV/bin:$PATH
WORKDIR /home/workdir

RUN pip install ninja yacs cython matplotlib tqdm opencv-python overrides ipython 
RUN pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html

ENV INSTALL_DIR=$PWD

RUN cd $INSTALL_DIR && \
    git clone https://github.com/cocodataset/cocoapi.git && \
    cd cocoapi/PythonAPI && \
    python setup.py build_ext install

RUN cd $INSTALL_DIR && \
    git clone https://github.com/NVIDIA/apex.git && \
    cd apex && \
    python setup.py install --cuda_ext --cpp_ext

RUN cd $INSTALL_DIR && \
    git clone https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch.git sg_benchmark && \
    cd sg_benchmark && \
    python setup.py build develop 


