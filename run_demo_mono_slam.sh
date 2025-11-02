#!/bin/bash

# DJI-     original 3840x2160, undistorted 3770x2120, build model using 1422x800
# Vulcan-  original 1936x1216, undistorted 1911x1200, build model using 1273x800

# Vulcan :
# Save Images/Times/dt/data to disk
# ./run_camera.sh
# ./get_frames.sh
# Save Inertial data to disk
# ./run_nav.sh
# ./get_inertial.sh
#
#export DATASET_PATH=/media/mrt/Whale/data/mapping/2024-07-22-16-21-36-orbit-/
export DATASET_PATH=/media/mrt/Whale/data/vulcan-mapping/2024-06-28-03-47-19-uotf-orbit-16/
#export DATASET_PATH=/media/mrt/Whale/data/mission-systems/2024_05_30_03_auto_orbit/
export WORKSPACE=$DATASET_PATH/orbslam
export MAX_DIMS=1900
export LIMAP_W=1900
export LIMAP_H=1200

# DJI :
#export DATASET_PATH=$PWD/data/DJI_153
#export WORKSPACE=$DATASET_PATH
#export MAX_DIMS=3840
#export LIMAP_W=3840
#export LIMAP_H=2120

# models and parameters
export ENGINE=../detr/src/window.engine
export LIMAP_CONFIG=cfgs/triangulation/default_fast.yaml # this is relative to limap

mkdir -p $WORKSPACE

run_orbslam=true
run_limap=false
run_tracking=false

# Run ORBSLAM
if [ "$run_orbslam" = true ]; then

    cd ORB_SLAM3/
    
	# Vulcan CSV demo (not ready yet)
	./Examples/Monocular/mono_vulcan \
	    Vocabulary/ORBvoc.txt \
	    Examples/Monocular/vulcan.yaml

	# Vulcan images demo (not ready yet)
	#./Examples/Monocular/mono_euroc \
	#	Vocabulary/ORBvoc.txt \
	#	Examples/Monocular/vulcan.yaml \
	#	$WORKSPACE/images \
	#	$WORKSPACE/times.txt

	# Vulcan images demo (not ready yet)
	#./Examples/Monocular/mono_geo_vulcan \
	#	Vocabulary/ORBvoc.txt \
	#	Examples/Monocular/vulcan.yaml \
	#	$WORKSPACE

	# Vulcan images demo (not ready yet)
	#./Examples/Monocular-Inertial/mono_inertial_euroc \
	#	Vocabulary/ORBvoc.txt \
	#	Examples/Monocular-Inertial/vulcan.yaml \
	#	$WORKSPACE \
	#	$WORKSPACE/cam0/times.txt

	# DJI images demo (has been tested and is working)
	#./Examples/Monocular/mono_euroc \
	#	Vocabulary/ORBvoc.txt \
	#	Examples/Monocular/dji.yaml \
	#	$DATASET_PATH/images/ \
	#	$DATASET_PATH/../timestamps/DJI_153_times.txt

	mkdir -p $WORKSPACE/sparse
	mv sparse/* $WORKSPACE/sparse

    cd ../
    
fi

# Run LIMAP
#source /home/mrt/anaconda3/etc/profile.d/conda.sh
#conda activate limap
if [ "$run_limap" = true ]; then

    cd limap/
    
	python3 runners/colmap_triangulation.py -c $LIMAP_CONFIG -a $WORKSPACE --output_dir $WORKSPACE --max_image_dim $MAX_DIMS
	python3 visualize_3d_lines.py --input_dir $WORKSPACE/finaltracks/
	
	cd ../
	
fi

# Track windows and compute normals
# TODO output window data to disk (location+normal)
if [ "$run_tracking" = true ]; then

	python3 get_windows.py $WORKSPACE --limap_w=$LIMAP_W --limap_h=$LIMAP_H --engine=$ENGINE # 1911x1200 (1936x1216) Vulcan, 3770x2120 (3840x2160) DJI
	
fi

echo Done ...
