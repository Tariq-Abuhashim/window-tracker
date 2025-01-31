FROM missionsystems/base:20.04
# FROM missionsystems/base:20.04-ros2-cuda-devel
# FROM ubuntu:20.04
USER root
RUN apt-get install -y tree
USER devops
WORKDIR /home/devops/src
RUN git clone git@gitlab.com:missionsystems/hyperteaming/window-tracker
WORKDIR window-tracker
RUN git submodule update --init --recursive || echo blah
RUN tree

