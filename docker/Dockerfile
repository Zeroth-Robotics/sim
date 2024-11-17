# 1stly, to build zeroth-bot-sim:v0
# FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04   

# COPY sources.list /etc/apt/sources.list

# RUN mkdir /root/.pip
# COPY pip.conf /root/.pip/pip.conf

# RUN apt install -y wget zlib1g-dev libssl-dev libncurses5-dev libsqlite3-dev libreadline-dev libtk8.6 libgdm-dev libdb4o-cil-dev libpcap-dev
# RUN wget https://mirrors.huaweicloud.com/python/3.8.19/Python-3.8.19.tar.xz && tar -xvf Python-3.8.19.tar.xz

# WORKDIR  /root/Python-3.8.19
# RUN ./configure --prefix=/usr/local && make && make install

# WORKDIR /usr/local/bin/
# RUN ln -s pip3 pip

# WORKDIR  /app/sim/
# RUN make install-dev

# RUN wget https://developer.nvidia.com/isaac-gym-preview-4
# # RUN tar -xvf 
# RUN make install-third-party-external

FROM zeroth-bot-sim:v0

WORKDIR /root

RUN apt update

# on vnc client use <your docker ip>:5901
RUN apt update && apt install -y \
    --no-install-recommends \
    libxcursor-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxi-dev \
    mesa-common-dev \
    zip \
    unzip \
    make \
    vulkan-utils \
    mesa-vulkan-drivers \
    pigz \
    git \
    libegl1

# WAR for eglReleaseThread shutdown crash in libEGL_mesa.so.0 (ensure it's never detected/loaded)
# Can't remove package libegl-mesa0 directly (because of libegl1 which we need)
RUN rm /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0 /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0.0.0 /usr/share/glvnd/egl_vendor.d/50_mesa.json

COPY ./nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY ./10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json


WORKDIR /app/sim

RUN useradd --create-home gymuser
USER gymuser

# copy gym repo to docker
COPY --chown=gymuser . .

# install gym modules
ENV PATH="/home/gymuser/.local/bin:$PATH"

RUN export MODEL_DIR=sim/resources/stompymicro

ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all

CMD ["tail","-f", "/dev/null"]

# CMD ["python3", "sim/train.py", "--task=stompymicro", "--num_envs=4"]
# python3 sim/train.py --task=stompymicro --num_envs=4

