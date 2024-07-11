#!/bin/bash

export DATASET_PATH=/media/mrt/Whale/data/mission-systems/2024-06-28-03-47-19-uotf-orbit-16/
export nav_port=4002
export BAG_NAME=*.bin

echo replay navigation data ...
cd $DATASET_PATH
nav_format=$( ms-log-multitool data --include advanced-navigation --output-format )
cat advanced-navigation/$BAG_NAME | ms-log-multitool data --include advanced-navigation | csv-play --binary $nav_format --slow 1 | io-publish tcp:$nav_port --size $( echo $nav_format | csv-format size )

echo Done !!!
