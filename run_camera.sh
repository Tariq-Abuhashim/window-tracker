#!/bin/bash

export DATASET_PATH=/media/mrt/Whale/data/mission-systems/2024-06-28-03-47-19-uotf-orbit-16/
export CAMERA=alvium_1800_down
export image_port=4001
export BAG_NAME=*.bin
image_format=t,3ui,s[7062528]

echo replay image data ...
cd $DATASET_PATH
cat cameras/$CAMERA/$BAG_NAME | csv-play --binary $image_format --slow 1 | io-publish tcp:$image_port --size $( echo $image_format | csv-format size )

echo Done !!!
