
#pip install pyproj
#pip install scipy

import os
import pandas as pd
import numpy as np
import pyproj
from scipy.spatial.transform import Rotation as R
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_gps_data(file_path):
    """Read GPS and orientation data from a space-separated CSV file."""
    # Load data with header correctly parsed
    data = pd.read_csv(file_path, delim_whitespace=True)

    # Create a dictionary to store the poses
    poses = {}
    print(f'Warning: yaw has been inverted by (-).')
    for index, row in data.iterrows():
        tx, ty, tz = gps_to_utm(row['lat'], row['lon'], row['alt'])
        x_colmap, y_colmap, z_colmap = utm_to_colmap(tx, ty, tz)
        rx, ry, rz = utm_to_colmap(-row['yaw'], row['pitch'], row['roll'])
        quaternion = euler_to_quaternion(rx, ry, rz)
        image_name = row['image']
        poses[image_name] = (x_colmap, y_colmap, z_colmap, quaternion)
    
    return poses

def utm_to_colmap(x, y, z):
    # Convert UTM East-North-Up to COLMAP Right-Back-Down
    return x, y, z  # x, -y, -z

def gps_to_utm(lat, lon, alt):
    """Convert GPS coordinates to UTM."""
    # Determine UTM zone
    utm_zone = int((lon + 180) / 6) + 1
    # Define the UTM projection
    # Including '+south' in the projection string for southern hemisphere handling
    utm_projection = pyproj.Proj(proj='utm', zone=utm_zone, ellps='WGS84', datum='WGS84', south=True)
    x, y = utm_projection(lon, lat)
    #print(f"Longitude: {lon}, Latitude: {lat}, Zone: {utm_zone}, Easting: {x}, Northing: {y}")
    return x, y, alt

def euler_to_quaternion(roll, pitch, yaw):
    """Convert Euler angles (in degrees) to quaternion."""
    #roll, pitch, yaw = np.deg2rad([roll, pitch, yaw])  # Convert degrees to radians
    rotation = R.from_euler('xyz', [roll, pitch, yaw])
    return rotation.as_quat()  # Returns (x, y, z, w)

def normalize_coordinates(poses):
    """ Normalize the coordinates of camera poses relative to the first camera pose. """
    # Assuming the first entry is the reference
    first_image_name = next(iter(poses))
    base_x, base_y, base_z = poses[first_image_name][0], poses[first_image_name][1], poses[first_image_name][2]

    # Normalize all coordinates
    normalized_poses = {}
    for image_name, (x, y, z, quaternion) in poses.items():
        norm_x = x - base_x
        norm_y = y - base_y
        norm_z = z - base_z
        normalized_poses[image_name] = (norm_x, norm_y, norm_z, quaternion)
    
    return normalized_poses

def plot_3d_poses(poses):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    tx, ty, tz = [], [], []
    directions = []
    
    for pose in poses.values():
        x, y, z, quaternion = pose
        tx.append(x)
        ty.append(y)
        tz.append(z)
        rot = R.from_quat(quaternion)
        directions.append(rot)

    tx = np.array(tx)
    ty = np.array(ty)
    tz = np.array(tz)
    
    scale_factor = 5 # Scale direction vectors for visibility
    
    # Pre-calculate direction vectors for all poses
    direction_vectors_x = [d.apply([1, 0, 0]) * scale_factor for d in directions]
    direction_vectors_y = [d.apply([0, 1, 0]) * scale_factor for d in directions]
    direction_vectors_z = [d.apply([0, 0, 1]) * scale_factor for d in directions]

    ax.scatter(tx, ty, tz, c='blue', label='Camera Position')
    for i in range(len(tx)):
        ax.quiver(tx[i], ty[i], tz[i], direction_vectors_x[i][0], direction_vectors_x[i][1], direction_vectors_x[i][2], color='red', label='X-axis' if i == 0 else "")
        ax.quiver(tx[i], ty[i], tz[i], direction_vectors_y[i][0], direction_vectors_y[i][1], direction_vectors_y[i][2], color='green', label='Y-axis' if i == 0 else "")
        ax.quiver(tx[i], ty[i], tz[i], direction_vectors_z[i][0], direction_vectors_z[i][1], direction_vectors_z[i][2], color='blue', label='Z-axis' if i == 0 else "")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Updated colmap priors')
    ax.legend()
    plt.show()



def update_database(db_path, poses):
    """Update the COLMAP database with new pose data."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    for image_name, (x, y, z, quaternion) in poses.items():
        #print(f"{image_name} {x} {y} {z} {quaternion}")
        cursor.execute("""
            UPDATE images
            SET prior_tx=?, prior_ty=?, prior_tz=?,
                prior_qw=?, prior_qx=?, prior_qy=?, prior_qz=?
            WHERE name=?""",
            (x, y, z, quaternion[3], quaternion[0], quaternion[1], quaternion[2], image_name))
    
    conn.commit()
    conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Update COLMAP database with GPS and orientation data.")
    parser.add_argument("--db_path", required=True, help="Path to the COLMAP database file.")
    parser.add_argument("--gps_data_file", required=True, help="Path to the GPS data CSV file.")
    args = parser.parse_args()

    # Read GPS data from CSV
    poses = read_gps_data(args.gps_data_file)

    # Normalize coordinates
    normalized_poses = normalize_coordinates(poses)

    # Plotting the normalized poses
    plot_3d_poses(normalized_poses)

    # Update database
    update_database(args.db_path, normalized_poses)

    print("Database has been updated with GPS poses.")
