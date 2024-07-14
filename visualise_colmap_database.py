import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import argparse

# python visualize_cameras.py --db_path "/path/to/your/colmap/database.db"

def visualize_colmap_poses(db_path):
    """Visualizes camera poses from a COLMAP database."""
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Execute query to get camera pose data
    cursor.execute("""
        SELECT prior_tx, prior_ty, prior_tz, prior_qw, prior_qx, prior_qy, prior_qz
        FROM images
        WHERE prior_tx IS NOT NULL AND prior_ty IS NOT NULL AND prior_tz IS NOT NULL
    """)
    data = cursor.fetchall()
    cursor.close()
    conn.close()

    if not data:
        print("No pose data available in the database.")
        return

    # Extract and convert data
    tx, ty, tz, qw, qx, qy, qz = zip(*data)
    tx, ty, tz = np.array(tx), np.array(ty), np.array(tz)
    quaternions = np.array([qw, qx, qy, qz]).T
    quaternions /= np.linalg.norm(quaternions, axis=1)[:, np.newaxis]  # Normalize quaternions

    # Create rotation objects from quaternions
    rotations = R.from_quat(quaternions)

    scale_factor = 5 # Scale direction vectors for visibility
    direction_vectors_x = rotations.apply([1, 0, 0]) * scale_factor
    direction_vectors_y = rotations.apply([0, 1, 0]) * scale_factor
    direction_vectors_z = rotations.apply([0, 0, 1]) * scale_factor

    # Visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the camera positions
    ax.scatter(tx, ty, tz, c='blue', label='Camera Position')

    # Plot arrows representing the camera orientation
    for i in range(len(tx)):
        ax.quiver(tx[i], ty[i], tz[i], direction_vectors_x[i, 0], direction_vectors_x[i, 1], direction_vectors_x[i, 2], color='red', label='X-axis' if i == 0 else "")
        ax.quiver(tx[i], ty[i], tz[i], direction_vectors_y[i, 0], direction_vectors_y[i, 1], direction_vectors_y[i, 2], color='green', label='Y-axis' if i == 0 else "")
        ax.quiver(tx[i], ty[i], tz[i], direction_vectors_z[i, 0], direction_vectors_z[i, 1], direction_vectors_z[i, 2], color='blue', label='Z-axis' if i == 0 else "")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Camera Poses and Orientations')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize camera poses from a COLMAP database.")
    parser.add_argument("--db_path", required=True, help="Path to the COLMAP database file.")
    args = parser.parse_args()

    visualize_colmap_poses(args.db_path)
