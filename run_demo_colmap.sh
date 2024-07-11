#!/bin/bash

# DJI-     original 3840x2160, rectified 3770x2120, build model using 1422x800
# Vulcan-  original 1936x1216, rectified 1911x1200, build model using 1273x800

#export DATASET_PATH=/media/mrt/Whale/data/mission-systems/2024-06-28-03-47-19-uotf-orbit-16/
#export DATASET_PATH=/media/mrt/Whale/data/mission-systems/2024_05_30_03_auto_orbit/
export DATASET_PATH=$PWD/data/DJI_78
export WORKSPACE=$DATASET_PATH

export ENGINE=../detr/src/window.engine
export LIMAP_CONFIG=cfgs/triangulation/default_fast.yaml # this is relative to limap
export MAX_DIMS=3840 # DJI-3840, Vulcan-1936

mkdir -p $WORKSPACE

run_colmap=true
run_limap=true

# Run COLMAP
cd ../colmap/build/
#colmap automatic_reconstructor --image_path ${IMAGES} --workspace_path ${WORKSPACE}
# export the model as text files, then check intrinsics in cameras.txt
if [ "$run_colmap" = true ]; then
	colmap feature_extractor \
	   --database_path $WORKSPACE/database.db \
	   --image_path $DATASET_PATH/images

	colmap exhaustive_matcher \
	   --database_path $WORKSPACE/database.db

	mkdir $WORKSPACE/sparse

	colmap mapper \
	   --database_path $WORKSPACE/database.db \
	   --image_path $DATASET_PATH/images \
	   --output_path $WORKSPACE/sparse

	mv $WORKSPACE/sparse/0/* $WORKSPACE/sparse
    rm -r $WORKSPACE/sparse/0
fi

# Run LIMAP
cd ../../window-tracker/limap
source /home/mrt/anaconda3/etc/profile.d/conda.sh
conda activate limap
if [ "$run_limap" = true ]; then
	python3 runners/colmap_triangulation.py -c $LIMAP_CONFIG -a $WORKSPACE --output_dir $WORKSPACE --max_image_dim $MAX_DIMS
	python3 visualize_3d_lines.py --input_dir $WORKSPACE/finaltracks/
fi

# Track windows and compute normals
# TODO output window data to disk (location+normal)
cd ../
python3 get_windows.py $WORKSPACE --limap_w=3770 --limap_h=2120 --engine=$ENGINE  # Rectified image dimensions: 1911x1200 Vulcan, 3770x2120 DJI
echo Done ...
