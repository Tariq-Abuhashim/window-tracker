FROM missionsystems/base:20.04-ros2-cuda-devel


# Create necessary directories and set workdir
RUN mkdir -p /home/devops/Thirdparty /home/devops/src/hyperteaming
WORKDIR /home/devops/Thirdparty


# Install OpenCV from apt repository
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-dev \
    python3-opencv


# Install PyTorch from source
USER devops
WORKDIR /home/devops/src
RUN git clone --recursive -b v1.11.0 https://github.com/pytorch/pytorch
WORKDIR /home/devops/src/pytorch
RUN git submodule sync && git submodule update --init --recursive --jobs 0
USER root
RUN python3 -m pip install astunparse numpy==1.19.5 ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
RUN CMAKE_CUDA_ARCHITECTURES="75" \
    python3 setup.py install


# Install TorchVision from source
USER devops
WORKDIR /home/devops/src
RUN git clone -b v0.12.0 https://github.com/pytorch/vision.git
WORKDIR /home/devops/src/vision
USER root
RUN python3 setup.py install


# Dependencies for CERES and COLMAP
USER root
RUN apt-get update
RUN apt-get install -y git cmake build-essential \
    libboost-program-options-dev libboost-filesystem-dev libboost-graph-dev \
    libboost-regex-dev libboost-system-dev libboost-test-dev \
    libeigen3-dev libsuitesparse-dev libfreeimage-dev libgoogle-glog-dev libgflags-dev libglew-dev \
    qtbase5-dev libqt5opengl5-dev libcgal-dev libsqlite3-dev\
    libatlas-base-dev libsuitesparse-dev libceres-dev libmetis-dev libhdf5-dev libflann-dev


# Install a specific version of CMake (if needed)
USER root
WORKDIR /home/devops/src
RUN wget https://github.com/Kitware/CMake/releases/download/v3.28.4/cmake-3.28.4-Linux-x86_64.sh \
    && chmod +x cmake-3.28.4-Linux-x86_64.sh \
    && ./cmake-3.28.4-Linux-x86_64.sh --skip-license


# Build and install Ceres Solver
#USER devops
#WORKDIR /home/devops/src
#RUN git clone https://ceres-solver.googlesource.com/ceres-solver
#WORKDIR /home/devops/src/ceres-solver
#RUN mkdir build
#WORKDIR /home/devops/src/ceres-solver/build
#RUN cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF
#RUN make -j$(nproc)
#USER root
#RUN make install


# Build and install COLMAP.
USER devops
WORKDIR /home/devops/src
RUN git clone -b 3.8 https://github.com/colmap/colmap.git
WORKDIR /home/devops/src/colmap
RUN git checkout 3.8
RUN git reset --hard 3.8
RUN echo "installing colmap ${PWD}"
RUN mkdir build
WORKDIR /home/devops/src/colmap/build
RUN cmake -DCMAKE_CUDA_ARCHITECTURES="75" ..
RUN make -j$(nproc)  # Compile using all available cores
USER root
RUN make install


# Enable GUI
USER root
RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
  && rm -rf /var/lib/apt/lists/*


# Install PoseLib
USER devops
WORKDIR /home/devops/src
RUN git clone --recursive https://github.com/vlarsson/PoseLib.git
WORKDIR /home/devops/src/PoseLib
RUN mkdir build
WORKDIR /home/devops/src/PoseLib/build
RUN cmake ..
RUN make -j$(nproc)
USER root
RUN make install


# Git LFS
USER root
RUN apt update
RUN apt install -y git-lfs

RUN echo 'new build'
# window-tracker
USER devops
WORKDIR /home/devops/src/hyperteaming

RUN git clone git@gitlab.com:missionsystems/hyperteaming/window-tracker.git
WORKDIR /home/devops/src/hyperteaming/window-tracker

RUN git submodule update --init --recursive || echo "cloned with errors"

#RUN git clone git@gitlab.com:missionsystems/hyperteaming/line-mapping.git limap

#WORKDIR /home/devops/src/hyperteaming/window-tracker/limap/third-party
#RUN git clone --recursive https://github.com/pybind/pybind11.git
#RUN git clone --recursive https://github.com/cvg/Hierarchical-Localization.git
#RUN git clone --recursive https://github.com/B1ueber2y/JLinkage
#RUN git clone --recursive https://github.com/B1ueber2y/libigl.git

# Set CMAKE_PREFIX_PATH to include PoseLib_DIR
##ENV CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}:/home/devops/src/PoseLib
#RUN git clone --recursive https://github.com/B1ueber2y/RansacLib.git
##WORKDIR /home/devops/src/hyperteaming/window-tracker/limap/third-party/RansacLib
##RUN mkdir build
##WORKDIR /home/devops/src/hyperteaming/window-tracker/limap/third-party/RansacLib/build
##RUN cmake -DCMAKE_PREFIX_PATH=/usr/local/lib/cmake/PoseLib ..
##RUN make -j$(nproc)
##USER root
##RUN make install

#WORKDIR /home/devops/src/hyperteaming/window-tracker/limap/third-party
#RUN git clone --recursive https://github.com/B1ueber2y/HighFive.git
#RUN git clone --recursive https://github.com/iago-suarez/pytlsd.git
#RUN git clone --recursive https://github.com/iago-suarez/pytlbd.git
#RUN git clone --recursive https://github.com/cherubicXN/hawp.git
#RUN git clone --recursive https://github.com/cvg/DeepLSD.git
#RUN git clone --recursive https://github.com/cvg/GlueStick.git

#RUN git clone https://github.com/rpautrat/TP-LSD.git
WORKDIR /home/devops/src/hyperteaming/window-tracker/limap/third-party/TP-LSD/tp_lsd/modeling
RUN git clone https://github.com/lucasjinreal/DCNv2_latest.git DCNv2

#WORKDIR /home/devops/src/hyperteaming/window-tracker/limap/third-party/pytlsd
#RUN python3 -m pip install .

#WORKDIR /home/devops/src/hyperteaming/window-tracker/limap/third-party/GlueStick
#RUN rm requirements.txt
#RUN printf 'from setuptools import setup\n\nsetup(name="gluestick", version="0.0", packages=["gluestick"])\n' > setup.py
#RUN python3 -m pip install .


#USER root

# Remove distutils installed PyYAML
#RUN apt-get remove -y python3-yaml

# Force reinstall PyYAML using pip to avoid distutils issues
#RUN python3 -m pip install --upgrade --force-reinstall pyyaml

#USER devops

WORKDIR /home/devops/src/hyperteaming/window-tracker/limap

RUN python3 -m pip install -r requirements.txt

#ENV CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}:/home/devops/src/PoseLib
ENV RansacLib_DIR=/home/devops/src/hyperteaming/window-tracker/limap/third-party/RansacLib

#WORKDIR /home/devops/src/hyperteaming/window-tracker/limap/third-party/RansacLib
#USER root
#RUN apt-get install -y moreutils
#USER devops
# todo! use git apply patch instead!
#RUN sed < CMakeLists.txt 's|add_subdirectory (examples)|#add_subdirectory (examples)|' | sponge CMakeLists.txt

#WORKDIR /home/devops/src/hyperteaming/window-tracker/limap

RUN python3 -m pip install .


# TensorRT and Pycuda
USER root
RUN apt-get update
RUN python3 -m pip install tensorrt
RUN apt install -y tensorrt
RUN apt install -y python3-pycuda
RUN apt install -y python3-tk


# DETR
WORKDIR /home/devops/src/hyperteaming
USER devops
RUN git clone git@gitlab.com:missionsystems/hyperteaming/detr.git
RUN pip3 install cython scipy
RUN pip3 install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
RUN pip3 install git+https://github.com/cocodataset/panopticapi.git


# switch user
USER devops
