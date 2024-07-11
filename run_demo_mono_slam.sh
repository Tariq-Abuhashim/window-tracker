#!/bin/bash

# DJI-     original 3840x2160, build model using 1422x800
# Vulcan-  original 1936x1216, build model using 1273x800

#export DATASET_PATH=/media/mrt/Whale/data/mission-systems/2024-06-28-03-47-19-uotf-orbit-16/
#export DATASET_PATH=/media/mrt/Whale/data/mission-systems/2024_05_30_03_auto_orbit/
export DATASET_PATH=$PWD/data/DJI_153
export WORKSPACE=$DATASET_PATH

export ENGINE=../detr/src/window.engine
export LIMAP_CONFIG=cfgs/triangulation/default_fast.yaml # this is relative to limap
export MAX_DIMS=3840 # DJI-3840, Vulcan-1936

mkdir -p $WORKSPACE

run_orbslam=true
run_limap=true

# Run ORBSLAM
cd ORB_SLAM3/
if [ "$run_orbslam" = true ]; then

	# Vulcan CSV demo (not ready yet)
	#./Examples/Monocular/mono_vulcan \
	#    Vocabulary/ORBvoc.txt \
	#    Examples/Monocular/vulcan.yaml

	# Vulcan images demo (not ready yet)
	#./Examples/Monocular/mono_euroc \
	#	Vocabulary/ORBvoc.txt \
	#	Examples/Monocular/vulcan.yaml \
	#	$WORKSPACE/images \
	#	$WORKSPACE/times.txt

	# DJI images demo (has been tested and is working)
	./Examples/Monocular/mono_euroc \
		Vocabulary/ORBvoc.txt \
		Examples/Monocular/dji.yaml \
		$DATASET_PATH/images/ \
		$DATASET_PATH/../timestamps/DJI_153_times.txt

	mkdir -p $WORKSPACE/sparse
	mv sparse/* $WORKSPACE/sparse

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
python3 get_windows.py $WORKSPACE --limap_w=3840 --limap_h=2160 --engine=$ENGINE # 1911x1200 (1936x1216) Vulcan, 3770x2120 (3840x2160) DJI
echo Done ...
