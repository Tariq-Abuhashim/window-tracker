#!/bin/bash

# Stop execution if any command fails
set -e

# Check DETR
python3 -c "import sys; sys.path.append('../detr/src'); from infer_engine import TensorRTInference"

# Check Python version
python3 --version
pip3 --version

# Update and Upgrade system packages
sudo apt-get update && sudo apt-get upgrade -y

# Install basic dependencies
sudo apt-get install -y cmake git libhdf5-dev build-essential

# Update pip
#python3 -m pip install --upgrade pip
#python3 -m pip install --upgrade pip setuptools

cd ../

# Install PyTorch compatible with your CUDA (Jetson doesnt support mkl)
# follow DETR steps, this is only a reinstall
# for Jetson, run the line without mkl and mkl-include
python3 -m pip install astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
#python3 -m pip install astunparse numpy ninja pyyaml setuptools cmake cffi typing_extensions future six requests dataclasses
git clone -b v1.12.0 https://github.com/pytorch/pytorch
cd pytorch
echo "installing pytorch ${PWD}"
sudo CMAKE_CUDA_ARCHITECTURES="75" CUDACXX=/usr/local/cuda/bin/nvcc python3 setup.py install

cd ../

# Install torchvision
# follow DETR steps, this is only a reinstall
git clone -b v0.13.0 https://github.com/pytorch/vision.git
cd vision
echo "installing torchvision ${PWD}"
sudo TORCH_CUDA_ARCH_LIST="7.5" python3 setup.py install

cd ../

# Clone COLMAP from the official repository
sudo apt-get update
sudo apt-get install -y git cmake build-essential \
    libboost-program-options-dev libboost-filesystem-dev libboost-graph-dev \
    libboost-regex-dev libboost-system-dev libboost-test-dev \
    libeigen3-dev libsuitesparse-dev libfreeimage-dev libgoogle-glog-dev libgflags-dev libglew-dev \
    qtbase5-dev libqt5opengl5-dev libcgal-dev libsqlite3-dev\
    libatlas-base-dev libsuitesparse-dev libceres-dev libmetis-dev libhdf5-dev libflann-dev
git -b v3.8 clone https://github.com/colmap/colmap.git
cd colmap
git checkout 3.8
git reset HEAD --hard
echo "installing colmap ${PWD}"
if [ ! -d "build" ]; then
    mkdir build
fi
cd build
cmake -DCMAKE_CUDA_ARCHITECTURES="75" ..
make -j$(nproc)  # Compile using all available cores
sudo make install
# sudo apt install -y colmap << THis is not a dev version, no cmakelists or headers.

cd ../../

# Clone and install PoseLib
git clone --recursive https://github.com/vlarsson/PoseLib.git
cd PoseLib
echo "installing PoseLib ${PWD}"
if [ ! -d "build" ]; then
    mkdir build
fi
cd build
cmake ..
make -j$(nproc)
sudo make install

cd ../../

# Build limap dependencies and install
git submodule update --init --recursive

cd limap
export LIMAP=$PWD
cd third-party/TP-LSD
git checkout 5558050
cd tp_lsd/modeling
rm -r DCNv2
git clone https://github.com/lucasjinreal/DCNv2_latest.git DCNv2
cd $LIMAP

python -m pip install -r requirements.txt
python -m pip install -Ive .

# Check if the limap package is installed
python3 -c "import limap" && echo "LIMAP package installed successfully" || echo "Failed to install LIMAP package"

# demo related, coordinate frame projections
python3 -m pip install pyproj

echo "Installation completed successfully."
