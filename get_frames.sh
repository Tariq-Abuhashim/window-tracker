#!/bin/bash

export DATASET_PATH=/media/mrt/Whale/data/mission-systems/2024-06-28-03-47-19-uotf-orbit-16/
export WORKSPACE=$DATASET_PATH/orbslam

mkdir -p $WORKSPACE
mkdir -p $WORKSPACE/images

export CAM_PORT=4001

# Run the image sampler
echo save images to disk ...
cd ../csv-to-euroc/
./build/bin/csv-to-images "t,3ui,s[7062528]" $CAM_PORT $WORKSPACE

# Repair red channel
#echo repair red channel ...
#cd ../repair-red/
#python3 repair_images.py $WORKSPACE/images --model red_channel_model.pkl

echo Done !!!
