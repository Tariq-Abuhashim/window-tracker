import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

def load_colmap_points(file_path):
    points = []
    colors = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue
            parts = line.split()
            x, y, z = map(float, parts[1:4])
            r, g, b = map(int, parts[4:7])
            points.append([x, y, z])
            colors.append([r / 255.0, g / 255.0, b / 255.0])
    return np.array(points), np.array(colors)

def load_normals(file_path):
    normals = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():
                nx, ny, nz = map(float, line.split())
                normals.append([nx, ny, nz])
    return normals

def load_camera_poses(file_path):
    poses = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            try:
                image_id = int(parts[0])
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                poses[image_id] = {'rotation': [qw, qx, qy, qz], 'translation': [tx, ty, tz]}
            except ValueError:
                continue
    return poses

def quaternion_to_rotation_matrix(q):
    q = np.array(q)
    if np.isclose(np.linalg.norm(q), 0):
        return np.eye(3)
    q /= np.linalg.norm(q)
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])

def visualize_points_normals_and_cameras(points, colors, normals, camera_poses, point_size=1, arrow_length=0.1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    points = np.array(points)
    normals = np.array(normals)

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=point_size)
    ax.quiver(points[:, 0], points[:, 1], points[:, 2], normals[:, 0], normals[:, 1], normals[:, 2], length=arrow_length, color='k', alpha=0.6)

    for pose in camera_poses.values():
        tx, ty, tz = pose['translation']
        R = quaternion_to_rotation_matrix(pose['rotation'])
        for i in range(3):
            ax.quiver(tx, ty, tz, R[i, 0]*arrow_length, R[i, 1]*arrow_length, R[i, 2]*arrow_length, color=['r', 'g', 'b'][i])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Points, Normals, and Camera Poses')

    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize COLMAP reconstruction with normals and camera poses.")
    parser.add_argument("points_path", help="Path to the COLMAP points3D.txt file")
    parser.add_argument("normals_path", help="Path to the normals results file")
    parser.add_argument("camera_poses_path", help="Path to the COLMAP images.txt file")

    args = parser.parse_args()

    #points, colors = load_colmap_points(args.points_path)
    #normals = load_normals(args.normals_path)
    camera_poses = load_camera_poses(args.camera_poses_path)

    points, colors, normals = [], [], []

    visualize_points_normals_and_cameras(points, colors, normals, camera_poses)

if __name__ == "__main__":
    main()
