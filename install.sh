#!/bin/bash

# Stop execution if any command fails
set -e

# Update and Upgrade system packages
sudo apt-get update && sudo apt-get upgrade -y

# Install basic dependencies
sudo apt-get install -y cmake git libhdf5-dev build-essential

# Install Python 3.9 and pip if not already installed
#conda create -n limap python=3.9 cudatoolkit=11.4
#conda activate limap 

# Check Python version
python3 --version
pip3 --version

# Update pip
python3 -m pip install --upgrade pip

# Install PyTorch compatible with your CUDA
pip install astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
git clone -b v1.12.0 https://github.com/pytorch/pytorch
cd pytorch
python3 setup.py install

#install torchvision
git clone -b v0.13.0 https://github.com/pytorch/vision.git
cd vision
python3 setup.py install

# Clone COLMAP from the official repository
sudo apt-get install \
    git \
    cmake \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-regex-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev
sudo apt install libatlas-base-dev libsuitesparse-dev
sudo apt install libceres-dev
sudo apt install libmetis-dev
sudo apt install libhdf5-dev
git clone https://github.com/colmap/colmap.git
cd colmap
git checkout tags/3.8 -b v3.8
mkdir build && cd build
cmake -DCMAKE_CUDA_ARCHITECTURES="75" ..
make -j$(nproc)  # Compile using all available cores
sudo make install
cd ../..

# Clone and install PoseLib
git clone --recursive https://github.com/vlarsson/PoseLib.git
cd PoseLib
mkdir build && cd build
cmake ..
make -j$(nproc)
cd ../..

# Initialize and update git submodules
#git clone https://github.com/cvg/limap.git

# Install LIMAP Python package
#cd limap
#git submodule update --init --recursive
# Install other Python packages from requirements.txt
#python3.9 -m pip install -r requirements.txt
#cd window-tracker/limap
#python3.9 setup.py install

# Check if the LIMAP package is installed
#python3.9 -c "import limap" && echo "LIMAP package installed successfully" || echo "Failed to install LIMAP package"

#echo "Installation completed successfully."