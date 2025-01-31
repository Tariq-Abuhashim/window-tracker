#!/bin/bash

# DJI-     original 3840x2160, undistorted 3770x2120, build model using 1422x800
# Vulcan-  original 1936x1216, undistorted 1911x1200, build model using 1273x800

# Vulcan :
# Save geo-referenced images/Times/dt to disk
# ./run_camera.sh
# ./run_nav.sh
# ./get_nav_image.sh
#
export DATASET_PATH=/media/mrt/Whale/data/mission-systems/2024-06-28-03-47-19-uotf-orbit-16/
# export DATASET_PATH=/mnt/orac-share/datasets/field-trials/hyperteaming/2024-06-singleton/vulcan/2024-06-28-03-47-19-uotf-orbit-16/
#export DATASET_PATH=/media/mrt/Whale/data/mission-systems/2024_05_30_03_auto_orbit/
export WORKSPACE=$DATASET_PATH/colmap_down
# export WORKSPACE=/home/fletcher/datasets/limap/demo
export MAX_DIMS=1936
export LIMAP_W=1911
export LIMAP_H=1200

# DJI :
#export DATASET_PATH=$PWD/data/DJI_153
#export WORKSPACE=$DATASET_PATH
#export MAX_DIMS=3840
#export LIMAP_W=3770
#export LIMAP_H=2120

# models and parameters
export ENGINE=../detr/src/window.engine
export LIMAP_CONFIG=cfgs/triangulation/default_fast.yaml # this is relative to limap

mkdir -p $WORKSPACE

run_colmap=false
run_limap=false
run_tracking=true
#run_colmap=true
#run_limap=true
#run_tracking=false


# Run COLMAP
#colmap automatic_reconstructor --image_path ${IMAGES} --workspace_path ${WORKSPACE}
# export the model as text files, then check intrinsics in cameras.txt
if [ "$run_colmap" = true ]; then
    # 0.183 minutes
	colmap feature_extractor \
	   	--database_path $WORKSPACE/database.db \
	   	--image_path $WORKSPACE/images

    # 5.900 minutes
	colmap exhaustive_matcher \
       	--database_path $WORKSPACE/database.db

    # x-right, y-forward, z-up (UTM - meters)
    python3 update_colmap.py \
       	--db_path $WORKSPACE/database.db \
       	--gps_data_file $WORKSPACE/nav/data.txt 

    # UTM to camera transform

    # x-right, y-backward, z-down (COLMAP frame - meters)
    python3 visualise_colmap_database.py \
       	--db_path $WORKSPACE/database.db
 
    # Check if database has been updated
    #sqlite3 $WORKSPACE/database.db
    #.headers on
    #.mode column
    #SELECT image_id, name, prior_tx, prior_ty, prior_tz, prior_qw, prior_qx, prior_qy, prior_qz FROM images;
    #SELECT image_id, name, tx, ty, tz, qw, qx, qy, qz FROM images;

	mkdir $WORKSPACE/sparse

    # 25.000 minutes
	colmap mapper \
	   	--database_path $WORKSPACE/database.db \
	   	--image_path $WORKSPACE/images \
	   	--output_path $WORKSPACE/sparse
		#--Mapper.priors_path $WORKSPACE/nav/data.txt

    mkdir $WORKSPACE/sparse/geo-registered-model

    # Batch align to reference gps
    # output: ecef or enu
	colmap model_aligner \
       --input_path $WORKSPACE/sparse/0 \
       --output_path $WORKSPACE/sparse/geo-registered-model \
       --ref_images_path $WORKSPACE/nav/gps.txt \
       --ref_is_gps 1 \
       --alignment_type enu \
       --robust_alignment 1 \
       --robust_alignment_max_error 3.0 #(where 3.0 is the error threshold to be used in RANSAC)

    # Move to sparse, because line-mapping looks there
    mv $WORKSPACE/sparse/geo-registered-model/* $WORKSPACE/sparse
    #mv $WORKSPACE/sparse/0/* $WORKSPACE/sparse
    #rm -r $WORKSPACE/sparse/0

	# Convert to text format, for easier access
    colmap model_converter \
      	--input_path $WORKSPACE/sparse/ \
      	--output_path $WORKSPACE/sparse \
      	--output_type TXT
fi

# Run LIMAP
cd limap
source /home/mrt/anaconda3/etc/profile.d/conda.sh
conda activate limap
# if facing issues linking libJLinkage.so
# export LD_LIBRARY_PATH= third-party/JLinkage/lib/:$LD_LIBRARY_PATH
if [ "$run_limap" = true ]; then
	python3 runners/colmap_triangulation.py -c $LIMAP_CONFIG -a $WORKSPACE --output_dir $WORKSPACE --max_image_dim $MAX_DIMS
	python3 visualize_3d_lines.py \
            --input_dir $WORKSPACE/finaltracks/ \
            --imagecols $WORKSPACE/imagecols.npy \
            --use_robust_ranges
            #--metainfos $WORKSPACE/metainfos
fi

# Track windows and compute normals
cd ../
if [ "$run_tracking" = true ]; then
	python3 get_windows.py $WORKSPACE --limap_w=$LIMAP_W --limap_h=$LIMAP_H --engine=$ENGINE  # Rectified image dimensions: 1911x1200 Vulcan, 3770x2120 DJI

	cd limap

	python3 visualize_3d_lines_and_window_normals.py \
		--input_dir $WORKSPACE/finaltracks/ \
		--imagecols $WORKSPACE/imagecols.npy \
		--normals_file $WORKSPACE/normals_results.json

    cd ../

    # convert window location from UTM (meters) to GPS (lat,lon, alt)
	python3 convert_normal_locations.py \
        --windows_file $WORKSPACE/normals_results.json \
        --reference_file $WORKSPACE/nav/gps.txt
fi

#cd ../../window-tracker
#python3 visualise_map_with_normals.py \
#	--points_path $WORKSPACE/sparse/points3D.txt
#	--camera_poses_path $WORKSPACE/sparse/images.txt
#	--normals_path $WORKSPACE/result_normals.txt

echo Done ...
