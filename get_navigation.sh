#!/bin/bash

export DATASET_PATH=/media/mrt/Whale/data/mission-systems/2024-06-28-03-47-19-uotf-orbit-16/
export WORKSPACE=$DATASET_PATH/orbslam
export NAV=$DATASET_PATH/orbslam/nav

mkdir -p $WORKSPACE
mkdir -p $NAV

export nav_port=4002

# Get format
export CURRENT_DIR=$PWD
cd $DATASET_PATH
nav_format=$( ms-log-multitool data --include advanced-navigation --output-format )

# Run the image sampler
echo save inertial to disk ...
cd $CURRENT_DIR/../csv-to-euroc/
./build/bin/csv-to-nav $nav_format 4002 $WORKSPACE

echo Done !!!
