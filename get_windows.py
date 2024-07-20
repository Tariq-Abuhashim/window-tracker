import os, sys
import argparse
import json
import cv2
import numpy as np
import open3d as o3d
from sklearn.linear_model import RANSACRegressor
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as T

sys.path.append("../detr/src")
from infer_engine import TensorRTInference
from window_tracker import Window, WindowTracker
from line_tracker import LineTracker

DEBUG = False

def data_loader(src_directory):
    dataset = []
    for filename in os.listdir(src_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):  # Image files
            image_path = os.path.join(src_directory, filename)
            # Assuming the camera file shares the same base name but with a '.P' extension
            camera_basename = os.path.splitext(filename)[0] + '.P'
            camera_path = os.path.join(src_directory, camera_basename)
            # Only add to dataset if both image and camera file exist
            if os.path.exists(camera_path):
                dataset.append((image_path, camera_path))
    return dataset

######################### normals clustering and refinement

from scipy.spatial import Delaunay

def delaunay_triangulation(locations):
    tri = Delaunay(locations)
    return tri

from collections import defaultdict

def cluster_normals_by_triangulation(tri, normals, locations, angle_threshold=0.1):
    """
    Cluster normals based on Delaunay Triangulation of their locations.

    :param tri: Delaunay triangulation of locations.
    :param normals: Normal vectors.
    :param locations: Locations associated with the normal vectors.
    :param angle_threshold: Threshold to consider normals as belonging to the same cluster.
    :return: Cluster labels for each normal.
    """
    def angle_between(v1, v2):
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    clusters = defaultdict(list)
    cluster_id = 0

    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                if angle_between(normals[simplex[i]], normals[simplex[j]]) < angle_threshold:
                    clusters[cluster_id].append(simplex[i])
                    clusters[cluster_id].append(simplex[j])
        cluster_id += 1

    labels = np.zeros(len(normals), dtype=int) - 1
    for cluster_id, indices in clusters.items():
        for idx in indices:
            labels[idx] = cluster_id

    return labels

def normalize_vectors(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

def refine_normals_by_cluster(normals, labels):
    """
    Refine normals within each cluster to point in the same direction.

    :param normals: A list of normal vectors.
    :param labels: Cluster labels for each normal.
    :return: A list of refined normal vectors.
    """
    unique_labels = np.unique(labels)
    refined_normals = np.array(normals)

    for label in unique_labels:
        # Get the indices of normals belonging to the current cluster
        cluster_indices = np.where(labels == label)[0]
        if len(cluster_indices) == 0:
            continue

        # Calculate the average normal for the current cluster
        average_normal = np.mean(refined_normals[cluster_indices], axis=0)
        average_normal /= np.linalg.norm(average_normal)  # Normalize the average normal

        # Align all normals in the cluster to point in the same direction as the average normal
        for idx in cluster_indices:
            dot_product = np.dot(refined_normals[idx], average_normal)
            if dot_product < 0:
                refined_normals[idx] = -refined_normals[idx]

    return refined_normals

######################### line-box association functions

def do_lines_intersect(line1, box):
    """Check if a line segment (defined by two points) intersects with a box using the Liang-Barsky algorithm."""
    (x1, y1), (x2, y2) = line1
    left, bottom, right, top = box

    def clip(p, q, u1, u2):
        """Helper function for clipping"""
        if p == 0 and q < 0:
            return False, u1, u2  # Line is parallel and outside the clipping window
        if p < 0:
            r = q / p
            if r > u2:
                return False, u1, u2  # Line is completely outside
            elif r > u1:
                u1 = r  # Line is entering the clipping window
        elif p > 0:
            r = q / p
            if r < u1:
                return False, u1, u2  # Line is completely outside
            elif r < u2:
                u2 = r  # Line is exiting the clipping window
        return True, u1, u2

    dx = x2 - x1
    dy = y2 - y1
    u1, u2 = 0.0, 1.0

    valid, u1, u2 = clip(-dx, x1 - left, u1, u2)
    if not valid:
        return False

    valid, u1, u2 = clip(dx, right - x1, u1, u2)
    if not valid:
        return False

    valid, u1, u2 = clip(-dy, y1 - bottom, u1, u2)
    if not valid:
        return False

    valid, u1, u2 = clip(dy, top - y1, u1, u2)
    if not valid:
        return False

    return True

# Function to check if both points of the line are inside the box
def is_point_in_box(point, box):
    left, bottom, right, top = box
    return left <= point[0] <= right and bottom <= point[1] <= top

def check_line_box(bbox, line):
    """
    This function returns True if the line l1-l2 intersects with any side of the box defined by bbox.
    """
    b1, b2 = bbox[:2], bbox[2:]
    l1, l2 = line
    if is_point_in_box(l1, bbox) and is_point_in_box(l2, bbox):
        return True
    return do_lines_intersect(line, bbox)

######################### window tracking functions

# Dictionaries to store relationships and observations
windows = {}

# Dictionary for direct 3D line ID to window ID mapping
line_to_window_map = {}

# Function to check existence of a 3D line ID
def find_3d_line_in_windows(line_ids):
    """
    Counts the number of occurrences of each line ID in the provided list and returns the window ID
    with the most line IDs associated with it.

    :param line_ids: List of line IDs to check.
    :return: The window ID with the most line IDs if found, otherwise None.
    """
    count = {}  # Dictionary to count occurrences of window_ids

    # Iterate over each line_id provided in the list
    for line_id in line_ids:
        if line_id in line_to_window_map:
            window_id = line_to_window_map[line_id]
            if window_id is None:
                continue
            # Increment the count for the window_id
            if window_id not in count:
                count[window_id] = 0
            count[window_id] += 1

    # Print the count of lines per window_id for debugging
    print(f'    win:#lines tracks = {count}')

    # Determine the window_id with the maximum number of associated lines
    if count:
        max_window_id = max(count, key=count.get) # Get the window_id with the maximum count
        if count[max_window_id] > 0:
            return max_window_id

    return None

# Initialize the structure for storing window relationships with 3D lines and 2D line observations by image
def initialize_window_data(window_id):
    if window_id not in windows:
        windows[window_id] = {
            '3d_lines': set(),
            '2d_lines_by_image': {}
        }
        print(f'    created new window {window_id}')

# Example of how to add data to the structure
def add_lines_to_window(window_id, line_3d_id, image_id):
    # Initialize the window data if not already present
    initialize_window_data(window_id)

    # Add 3D lines to windows
    for line_id in line_3d_id:
        windows[window_id]['3d_lines'].add(line_id) # Add 3D line ID to the set (automatically handles duplicates)
        line_to_window_map[line_id] = window_id # Update the map for each line ID

    # Initialize the list for this image if it doesn't exist
    #if image_id not in windows[window_id]['2d_lines_by_image']:
    windows[window_id]['2d_lines_by_image'][image_id] = []

    # Add the 2D line ID to the list for this image
    #windows[window_id]['2d_lines_by_image'][image_id].append(line_2d_id)

######################### normals estimation functions

# Compute normals for the 3D lines associated with a window
def compute_normals(window_id, tracker):
    center, normals = [], []
    # Use get() to avoid KeyError if window_id is not in the dictionary
    window_data = windows.get(window_id)

    if window_data:
        # Retrieve the set of 3D lines associated with the window
        lines_3d = [tracker.get_3d_line(i) for i in window_data['3d_lines']]
        center, normals = compute_3d_normals(lines_3d)
    else:
        print(f"No data available for window ID {window_id}")

    return center, normals, lines_3d

def compute_3d_normals(lines_3d):
    # First, compute the normal vector from the 3D lines
    normal, plane_point = fit_plane_to_lines_ransac(lines_3d)
    
    # Then calculate the center, ensuring it's projected onto the plane
    center = calculate_center_on_plane(lines_3d, normal, plane_point)
    
    return center, normal

def fit_plane_to_lines_ransac(lines):
    # Flatten line endpoints into an array of points and sample more points along each line
    points = []
    for line in lines:
        start_point, end_point = line
        sampled_points = sample_points_on_line(np.array(start_point), np.array(end_point), 20)
        points.extend(sampled_points)
    points = np.array(points)
    
    # Choose coordinates for fitting (here using x and z)
    xz = points[:, [0, 2]]  # Select the x and z coordinates
    y = points[:, 1]        # Select the y coordinates, assuming vertical variation in x-z plane
    
    # Fit a plane using RANSAC in the x-z plane
    ransac = RANSACRegressor(residual_threshold=0.01)
    ransac.fit(xz, y)

    # Extract the coefficients for the plane y = ax + cz + d
    a, c = ransac.estimator_.coef_
    d = ransac.estimator_.intercept_
    normal = np.array([-a, 1, -c])  # Normal to the plane
    
    # Normalize the normal vector
    normal /= np.linalg.norm(normal)
    
    # Use the mean of the points as a point on the plane
    plane_point = np.mean(points, axis=0)
    
    return normal, plane_point

def calculate_center_on_plane(lines, normal, plane_point):
    # Project points onto the plane and calculate their center
    points = np.array([point for line in lines for point in line])
    points_projected = np.array([project_point_onto_plane(point, normal, plane_point) for point in points])
    center = np.mean(points_projected, axis=0)
    return tuple(center)

def project_point_onto_plane(point, normal, plane_point):
    # Calculate the vector from the point on the plane to the point
    point_vector = point - plane_point
    # Calculate the distance from the point to the plane along the normal
    distance = np.dot(point_vector, normal)
    # Subtract the normal component from the point
    projected_point = point - distance * normal
    return projected_point

def create_line_set(lines, color):
    points = [point for line in lines for point in line]
    lines_indices = [[i, i+1] for i in range(0, len(points), 2)]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines_indices),
    )
    line_set.colors = o3d.utility.Vector3dVector([color] * (len(points) // 2))
    return line_set

def create_arrow(center, normal):
    arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.005,
                                                   cone_radius=0.01,
                                                   cylinder_height=0.1,
                                                   cone_height=0.02)
    arrow.paint_uniform_color([1, 0, 0])  # Red color for the normal vector
    arrow.translate(center)  # Move arrow to the center
    arrow.scale(2, center=center)  # Scale the arrow
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz(normal)
    arrow.rotate(rotation_matrix, center=center)
    return arrow

def create_plane_mesh(normal, center, scale=10, size=10):
    """
    Create a mesh for a plane given a normal vector and a point on the plane.
    The plane is centered at the given point and scaled by the scale factor.
    """
    # Define the mesh grid
    x = np.linspace(-scale, scale, size) + center[0]
    y = np.linspace(-scale, scale, size) + center[1]
    xx, yy = np.meshgrid(x, y)
    
    # Calculate zz using the plane equation ax + by + cz + d = 0
    d = -np.dot(normal, center)
    zz = (-normal[0] * xx - normal[1] * yy - d) / normal[2]
    
    # Create vertices and triangles for the mesh
    vertices = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    triangles = []
    for i in range(len(x) - 1):
        for j in range(len(y) - 1):
            idx = i * len(y) + j
            triangles.extend([[idx, idx + len(y), idx + 1],
                              [idx + 1, idx + len(y), idx + len(y) + 1]])
    # Create the mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    return mesh

def visualize_with_open3d(lines, normal, point_on_plane):
    # Create a visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add lines to the visualizer
    for line in lines:
        points = np.array(line)
        lines = [[0, 1]]
        colors = [[1, 0, 0]]  # Red color for the lines
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(line_set)

    # Add the plane mesh to the visualizer
    plane_mesh = create_plane_mesh(normal, point_on_plane)
    plane_mesh.paint_uniform_color([0.5, 0.5, 0.5])  # Grey color for the plane
    vis.add_geometry(plane_mesh)

    # Add the normal as an arrow
    arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.1, cone_radius=0.2, cylinder_height=2, cone_height=0.5)
    arrow.translate(point_on_plane)
    arrow.rotate(o3d.geometry.get_rotation_matrix_from_xyz(normal), center=point_on_plane)
    arrow.paint_uniform_color([0, 1, 0])  # Green color for the normal
    vis.add_geometry(arrow)

    # Run the visualizer
    vis.run()
    vis.destroy_window()

def sample_points_on_line(start_point, end_point, num_samples):
    """
    Linearly sample points between start and end points of a line segment.
    
    :param start_point: The starting point of the line (numpy array or list of coordinates).
    :param end_point: The ending point of the line (numpy array or list of coordinates).
    :param num_samples: Number of points to sample along the line.
    :return: A numpy array of shape (num_samples, 3) containing points along the line.
    """
    # Create a vector of sampled points along the line
    return np.linspace(start_point, end_point, num_samples)

def lines_to_point_cloud(lines, num_samples_per_line=10):
    """
    Converts a list of lines into a point cloud by sampling points along each line.
    
    :param lines: A list of tuples, each containing the start and end points of a line.
    :param num_samples_per_line: Number of points to sample along each line.
    :return: A numpy array of points.
    """
    points = []
    for line in lines:
        start_point, end_point = line
        points.append(sample_points_on_line(start_point, end_point, num_samples_per_line))
    return np.vstack(points)  # Stack all sampled points into a single array

def estimate_normals(points):
    """
    Estimates normals for the given set of points using Open3D.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    
    # Optionally orient normals (if you have a viewpoint or consistent orientation requirement)
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0, 0, 0]))
    
    return pcd

def rotation_matrix_from_vectors(vec1, vec2):
    """Create a rotation matrix that aligns vec1 to vec2."""
    a, b = (vec1 / np.linalg.norm(vec1)), (vec2 / np.linalg.norm(vec2))
    v = np.cross(a, b)
    c = np.dot(a, b)
    if c == -1:
        # Vectors are opposite
        c = np.max(vec1)
        idx = np.where(vec1 == c)[0][0]
        v = np.zeros_like(vec1)
        v[(idx+1)%3] = 1  # Use a perpendicular vector
    elif c == 1:
        # Vectors are the same
        return np.identity(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + k + np.dot(k, k) * ((1 - c) / (np.linalg.norm(v)**2))
    return R

# Update window with normal and center
def update_window(window_id, normal, center):
    if window_id in windows:
        window_data = windows.get(window_id, {})
        window_data['center'] = center
        window_data['normal'] = normal
        windows[window_id] = window_data  # Update the dictionary with the new data
        print(f'    Updated window {window_id} with normal and center')
    else:
        print(f'    Window {window_id} doest exist')

# Save windows to file
def write_windows(filename, line_tracker):
    """
    Write the results to a file in JSON format.

    :param filename: Name of the file to write the results.
    :param windows: Dictionary containing window data.
    """
    def convert_to_list(obj):
        """Helper function to convert numpy arrays and other objects to JSON-serializable lists."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [convert_to_list(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: convert_to_list(v) for k, v in obj.items()}
        else:
            return obj

    data_to_write = []
    for window_id, window_data in windows.items():
        center = window_data.get('center')
        normal = window_data.get('normal')
        lines = window_data.get('3d_lines', [])
        line_coords = [convert_to_list(line_tracker.get_3d_line(line_id)) for line_id in lines]
        num_lines = len(lines)
        
        if center is not None and normal is not None:
            record = {
                "window_id": window_id,
                "normal_location": convert_to_list(center),
                "normal_direction": convert_to_list(normal),
                "number_lines": num_lines,
                "lines_coordinates": line_coords
            }
            data_to_write.append(record)
        else:
            print(f'Window {window_id} does not have center or normal')

    with open(filename, 'w') as file:
        json.dump(data_to_write, file, indent=4)

def main(args):

    # Structure to hold estimated normals for each tracked window across images
    all_estimated_normals = {}

    # Initialise TensorRTInference with the tensorRT model
    trt_inference = TensorRTInference(args.engine, 1)#num_classes=1
    print("Detector Ready ...")

    # Initialise LineTracker
    line_tracker = LineTracker(args.work_dir+'/finaltracks/')
    print("Line Ready ...")

    window_counter = 0
    num_images = line_tracker.get_number_of_images()
    num_tracks = line_tracker.get_num_tracks()

    # Check the camera intrinsics (colmap each image is camera, slam 1 camera)
    #for image_id in range(1, num_images):
    #    K = line_tracker.get_intrinsic_matrix(image_id)
    #    if K is not None:
    #        print(line_tracker.get_intrinsic_matrix(image_id))

    for image_id in range(1, num_images):
        print(f'Keyframe {image_id}')

        # Get an image
        # for DJI dataset:
        #    original	3840x2160x3
        #    limap		3770x2120x3  (average undistorted)
        #    Onnx model	1422x800 x3
        # for Vulcan dataset:
        #    original	1936x1216x3
        #    limap		1911x1200x3  (average undistorted)
        #    Onnx model	1274x800 x3

        image_path = line_tracker.get_image_name(image_id)
        if DEBUG:
            print(image_path)
        current_image = cv2.imread(image_path)
        if current_image is None:
            continue
        h, w, c = current_image.shape
        if h != args.limap_h or w != args.limap_w:
            current_image = cv2.resize(current_image, (args.limap_w, args.limap_h))
        if DEBUG:
            print(current_image.shape)

        # Detect windows
        probas, bboxes = trt_inference.infer(current_image)
        detected_windows = [Window(bbox) for prob, bbox in zip(probas, bboxes) if np.argmax(prob) == 1] #FIXME # 3-for car,person, window,   1-for window only
        print(f'    number of detections {len(detected_windows)}')
        
        # Check track in window
        for detection in detected_windows:
            lines_in_current_detection = []
            for track_id in range(1, num_tracks): # equivalent to [1:num_tracks+1)

                # Check 3d line length
                if line_tracker.linetracks[track_id].line.length() > 3: #reject long lines (in meters)
                   continue

                # Detect 2D lines
                line2d = line_tracker.get_2d_line_in_image(track_id, image_id) # is there something that I can say, which tracks are in the current image
                #line2d = line_tracker.get_a_projection(track_id, image_id)
                if line2d is None:
                    continue
                line2d_array = line2d.as_array()
                #print(line2d_array)

                if check_line_box(detection.bounding_box, line2d_array):
                    lines_in_current_detection.append(track_id)
                    if DEBUG:
                        (x1, y1), (x2, y2) = line2d_array
                        cv2.line(current_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                else:
                    if DEBUG:
                        (x1, y1), (x2, y2) = line2d_array
                        cv2.line(current_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
            if DEBUG:
                print(f'    number of supporting lines = {len(lines_in_current_detection)}')

            window_id = 'x'
            if len(lines_in_current_detection) > 2: # was 3
                #print(lines_in_current_detection)
                window_id = find_3d_line_in_windows(lines_in_current_detection)
                if image_id == 1:
                    window_counter += 1
                    window_id = window_counter
                    add_lines_to_window(window_id, lines_in_current_detection, image_id)
                elif window_id is not None:
                    print(f'    updating window {window_id}')
                    add_lines_to_window(window_id, lines_in_current_detection, image_id)
                else:
                    window_counter += 1
                    window_id = window_counter
                    add_lines_to_window(window_id, lines_in_current_detection, image_id)
            if DEBUG:
                x1, y1, x2, y2 = detection.bounding_box
                cv2.rectangle(current_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)
                text = f"{window_id}"
                text_x = int(x1)
                text_y = int(y2) + 20 # 20 pixels below the bottom of the rectangle
                cv2.putText(current_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 1)
        if DEBUG:
            current_image = cv2.resize(current_image, (1600, 900))
            cv2.imshow('image', current_image)
            cv2.moveWindow('image', 2001, 100)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # Initialize visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Step 1: Compute normals and locations
    num_samples_per_line = 20
    all_normals = []
    all_locations = []
    for window_id, window_data in windows.items():
        center, normal, lines = compute_normals(window_id, line_tracker)  # Ensure this function returns correct values

        # Visualize the lines
        line_set = create_line_set(lines, np.random.rand(3))
        vis.add_geometry(line_set)

        # Sample more points along each line
        all_points = []
        for line in lines:
            start_point, end_point = line
            sampled_points = sample_points_on_line(np.array(start_point), np.array(end_point), num_samples_per_line)
            all_points.extend(sampled_points)

        # Create a point cloud from the sampled points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        
        # Estimate normals for the point cloud
        #pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=10))

        # Calculate the mean normal
        #normals = np.asarray(pcd.normals)
        #average_normal = np.mean(normals, axis=0)
        #average_normal /= np.linalg.norm(average_normal)  # Normalize the average normal

        # Compute the geometric center of the point cloud
        points = np.asarray(pcd.points)
        center_location = np.mean(points, axis=0)

        all_normals.append(normal)
        all_locations.append(center_location)

        # Update the window
        update_window(window_id, normal, center_location)

    # write window results to file
    write_windows(args.work_dir+"/normals_results.json", line_tracker)



    # Step 2: Normalize the normals
    normalized_normals = normalize_vectors(all_normals)

    # Step 3: Perform Delaunay Triangulation on locations
    tri = delaunay_triangulation(all_locations)

    # Step 4: Cluster normals based on the triangulation
    labels = cluster_normals_by_triangulation(tri, normalized_normals, all_locations, angle_threshold=0.1)

    # Step 5: Refine the normals based on the clusters
    refined_normals = refine_normals_by_cluster(normalized_normals, labels)

    # Debugging: Print unique labels and the color map size
    unique_labels = np.unique(labels)
    print(f"Unique labels: {unique_labels}")
    print(f"Number of clusters: {len(unique_labels)}")

    # Create a mapping from original labels to a range of indices
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

    # Map each label to a unique color
    color_map = plt.get_cmap("tab10")  # Use a colormap with enough distinct colors
    colors = [color_map(i / len(unique_labels))[:3] for i in range(len(unique_labels))]

    # Visualize the refined normals with colors based on clusters
    for normal, location, label in zip(refined_normals, all_locations, labels):
        print(f'{normal} {location}')
        if label in label_mapping:
            color = colors[label_mapping[label]]
        else:
            color = [1, 0, 0]  # Default to red if label is not found in mapping

        arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.2, cone_radius=0.4, cylinder_height=2, cone_height=1.0)
        transform = np.eye(4)
        transform[:3, 3] = location
        transform[:3, :3] = rotation_matrix_from_vectors(np.array([0, 0, 1]), normal)
        arrow.transform(transform)
        arrow.paint_uniform_color(color)  # Color based on cluster
        vis.add_geometry(arrow)
      
    # Run the visualizer
    vis.run()
    vis.destroy_window()

    # delete objects to avoid memory segmentation fault
    trt_inference.cleanup()

class Args:
    def __init__(self):
        self.data_dir = 'detr/data/dunster/'
        self.output_dir = 'detr/data/dunster/outputs/'
        self.limap_dir = 'limap/'
        self.detr_dir = 'detr/'

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(description='Tracks windows in images and computes their normals.')
        parser.add_argument('work_dir', type=str, help='Directory containing images with missing red channel')
        parser.add_argument('--engine', type=str, default='../detr/src/window.engine', help='Path to the trained model file')
        parser.add_argument('--limap_h', type=int, default=2120, help='Image height after undistort')
        parser.add_argument('--limap_w', type=int, default=3770, help='Image width after undistort')
        return parser.parse_args()

if __name__ == "__main__":
    inargs = Args.parse_args()
    args = Args()
    args.work_dir = inargs.work_dir
    args.engine = inargs.engine
    args.limap_h = inargs.limap_h
    args.limap_w = inargs.limap_w
    main(args)
