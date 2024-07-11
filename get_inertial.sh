#!/bin/bash

export DATASET_PATH=/media/mrt/Whale/data/mission-systems/2024_05_30_03_auto_orbit/
export WORKSPACE=$DATASET_PATH/orbslam
export IMU=$DATASET_PATH/orbslam/imu

mkdir -p $WORKSPACE
mkdir -p $IMU

export nav_port=4002

# Get format
export CURRENT_DIR=$PWD
cd $DATASET_PATH
nav_format=$( ms-log-multitool data --include advanced-navigation --output-format )

# Run the image sampler
echo save inertial to disk ...
cd $CURRENT_DIR/../csv-to-euroc/
./build/bin/csv-to-imu $nav_format 4002 $WORKSPACE

echo Done !!!
