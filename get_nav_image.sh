#!/bin/bash

export DATASET_PATH=/media/mrt/Whale/data/mission-systems/2024-06-28-03-47-19-uotf-orbit-16/
export WORKSPACE=$DATASET_PATH/orbslam
export NAV=$DATASET_PATH/orbslam/nav

mkdir -p $WORKSPACE
mkdir -p $NAV

export IMAGE_PORT=4001
export NAV_PORT=4002

# Get format
export CURRENT_DIR=$PWD
cd $DATASET_PATH
NAV_FORMAT=$( ms-log-multitool data --include advanced-navigation --output-format )
IMAGE_FORMAT="t,3ui,s[7062528]"

# Run the image sampler
echo save inertial to disk ...
cd $CURRENT_DIR/../csv-to-euroc/
./build/bin/csv-to-nav-images $IMAGE_FORMAT $IMAGE_PORT $NAV_FORMAT $NAV_PORT $WORKSPACE

echo Done !!!
