#
import os, sys
import cv2
import glob
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import open3d as o3d

import numpy as np
from sklearn.linear_model import RANSACRegressor


# conda activate detr
# cd /home/mrt/dev/limap/
# python setup.py build
# python setup.py install

sys.path.append("limap")
from line_tracker import LineTracker

sys.path.append("detr")
from detect_trt import TensorRTInference
from window_tracker import Window, WindowTracker

import torchvision.transforms as T

transform = T.Compose([
            T.Resize(900)
        ])

class Args:
    def __init__(self):
        self.data_dir = 'detr/data/dunster/'
        self.output_dir = 'detr/data/dunster/outputs/'
        self.limap_dir = 'limap/'
        self.detr_dir = 'detr/'


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

def do_lines_intersect(line1, box):
    """Check if a line segment (defined by two points) intersects with a box."""
    #print(line1)
    (x1, y1), (x2, y2) = line1
    line_box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    # Quick bounding box check to rule out no intersection possibility
    if (line_box[2] < box[0] or line_box[0] > box[2] or
            line_box[3] < box[1] or line_box[1] > box[3]):
        return False
    
    # Line clipping as per Cohen-Sutherland or Liang-Barsky can be implemented here for precise calculation
    
    # This example assumes intersection for simplicity if bounding boxes of the line and box overlap
    return True  # Replace with detailed algorithm or library function for actual geometric computation

def get_intersection(p1, p2, p3, p4):
    """
    This function checks if the line segment p1-p2 intersects with line segment p3-p4
    and returns the intersection point if it exists.
    """
    # Line p1-p2 represented as a1x + b1y = c1
    a1 = p2[1] - p1[1]
    b1 = p1[0] - p2[0]
    c1 = a1 * p1[0] + b1 * p1[1]

    # Line p3-p4 represented as a2x + b2y = c2
    a2 = p4[1] - p3[1]
    b2 = p3[0] - p4[0]
    c2 = a2 * p3[0] + b2 * p3[1]

    determinant = a1 * b2 - a2 * b1
    if determinant == 0:
        # The lines are parallel
        return None
    else:
        # The intersection point is given by (x, y)
        x = (b2 * c1 - b1 * c2) / determinant
        y = (a1 * c2 - a2 * c1) / determinant
        if min(p1[0], p2[0]) <= x <= max(p1[0], p2[0]) and min(p3[0], p4[0]) <= x <= max(p3[0], p4[0]):
            if min(p1[1], p2[1]) <= y <= max(p1[1], p2[1]) and min(p3[1], p4[1]) <= y <= max(p3[1], p4[1]):
                return (x, y)
        return None

# Define a function to check if a point is inside the bounding box
def is_point_in_box(point, b1, b2):
    return b1[0] <= point[0] <= b2[0] and b1[1] <= point[1] <= b2[1]

# Check if both points of the line are inside the box
def check_line_within_box(b1, b2, l1, l2):
    if is_point_in_box(l1, b1, b2) and is_point_in_box(l2, b1, b2):  # FIXME one point, or both?
        return True
    return False

def check_line_box(bbox, line):
    """
    This function returns True if the line l1-l2 intersects with any side of the box defined by b1 and b2.
    """
    b1 = bbox[0:2]
    b2 = bbox[2:4]
    l1 = line[0]
    l2 = line[1]

    # Early exit if at least one end-point is inside the box
    if check_line_within_box(b1, b2, l1, l2):
        return True

    box_lines = [
        (b1, (b2[0], b1[1])),  # Bottom side
        ((b2[0], b1[1]), b2),  # Right side
        (b2, (b1[0], b2[1])),  # Top side
        ((b1[0], b2[1]), b1)   # Left side
    ]

    #for box_line in box_lines:
    #    if get_intersection(l1, l2, box_line[0], box_line[1]):
    #        return True

    return False

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
        max_window_id = max(count, key=count.get)  # Get the window_id with the maximum count
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

# Compute normals for the 3D lines associated with a window
def compute_normals(window_id, tracker):
    center = []
    normals = []
    lines_3d = []
    # Use get() to avoid KeyError if window_id is not in the dictionary
    window_data = windows.get(window_id)

    if window_data:
        # Retrieve the set of 3D lines associated with the window
        lines_3d_id = window_data['3d_lines']
        # Placeholder for actual normal computation logic
        # Assuming you would calculate normals based on the 3D lines data
        # Here, you would implement the mathematical operations to compute normals
        lines_3d = []
        for i in lines_3d_id:
            lines_3d.append(tracker.get_3d_line(i))
        #print(lines_3d)
        center, normals = compute_3d_normals(lines_3d)
    else:
        print(f"No data available for window ID {window_id}")

    return center, normals, lines_3d

import numpy as np

def compute_3d_normals(lines_3d):
    # First, compute the normal vector from the 3D lines
    normal, plane_point = fit_plane_to_lines_ransac(lines_3d)
    
    # Then calculate the center, ensuring it's projected onto the plane
    center = calculate_center_on_plane(lines_3d, normal, plane_point)
    
    return center, normal

def fit_plane_to_lines(lines):
    # Prepare matrix A from line points
    A = np.array([[x, y, z, 1] for line in lines for point in line for x, y, z in (point,)])
    
    # Apply Singular Value Decomposition
    U, s, Vt = np.linalg.svd(A)
    
    # The plane coefficients are in the last row of Vt
    plane_coeffs = Vt[-1]
    normal = plane_coeffs[:3]
    d = plane_coeffs[3]
    
    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)
    
    # Return the normal and a point on the plane (we can use the mean of the points for simplicity)
    points_array = np.array([point for line in lines for point in line])
    plane_point = np.mean(points_array, axis=0)
    
    return normal, plane_point

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
    coef = ransac.estimator_.coef_
    a, c = coef
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

def main(args):

    # Structure to hold estimated normals for each tracked window across images
    all_estimated_normals = {}

    # Initialise TensorRTInference with the tensorRT model
    trt_inference = TensorRTInference('detr/detr.trt')
    print("Detector Ready ...")

    # Initialise LineTracker
    line_tracker = LineTracker('/media/mrt/Whale/data/mission_systems/window_tracker/data/DJI/finaltracks/')
    print("Line Ready ...")

    # Initialise WindowTracker with desired tracking method
    window_tracker = WindowTracker("iou", None)  # or "edges" for edge-line-based tracking
    print("Tracker Ready ...")

    # Load the dataset
    #dataset = data_loader(args.data_dir)
    #print("Data Loaded !!!!")
    #for image_path, camera_path in dataset:
        #I = cv2.imread(image_path)
        #P = np.loadtxt(camera_path)
        #print("Image shape:", I.shape)
        #print("Camera parameters shape:", P.shape
    
    window_counter = 0;
    num_images = line_tracker.get_number_of_images()
    num_tracks = line_tracker.get_num_tracks()

    for image_id in range(31, num_images):
        print(line_tracker.get_intrinsic_matrix(image_id))

    for image_id in range(1, num_images):

        print(f'Keyframe {image_id}')

        # Get an image
        # original size is 3840x2160x3
        # limap is using 1600x900x3  (scale=2.4, which can be adjusted in the config files)
        # Onnx model is using 1422x800x3 (scale=2.7)
        image_path = line_tracker.get_image_name(image_id)
        #print(image_path)
        current_image = cv2.imread(image_path)
        h, w, c = current_image.shape
        size = w*h
        if size!=23977200: #1440000: # resize to match limap
            current_image = cv2.resize(current_image, (3770, 2120))#(1600, 900)) # resize it to match limap size
        #print(current_image.shape)

        # Detect windows
        probas, bboxes = trt_inference.detect(current_image) # TODO link this to window detector
        detected_windows = [Window(bbox) for prob, bbox in zip(probas, bboxes) if np.argmax(prob) == 1] #FIXME # 3-for car,person, window,   1-for window only
        print(f'    number of detections {len(detected_windows)}')

        # Pose
        #current_pose = line_tracker.get_campose(image_id)
        #print(current_pose)

        # Track boxes
        #window_tracker.track_and_assign_ids(detected_windows, current_pose, intrinsic_matrix)

        # Check track in window
        for detection in detected_windows:
            lines_in_current_detection = []
            #print('-------')
            for track_id in range(1, num_tracks): # equivalent to [1:num_tracks+1)
    
                # Check 3d line length
                if (line_tracker.linetracks[track_id].line.length()>2):  # FIXME reject long lines
                   continue

                # Detect 2D lines
                line2d = line_tracker.get_2d_line_in_image(track_id, image_id) # FIXME is there something that I can say, which tracks are in the current image
                #line2d = line_tracker.get_a_projection(track_id, image_id)
                if line2d is None:
                    continue
                line2d_array = line2d.as_array()

                if check_line_box(detection.bounding_box, line2d_array):
                    lines_in_current_detection.append(track_id)
                    #(x1, y1), (x2, y2) = line2d_array
                    #cv2.line(current_image,(int(x1), int(y1)), (int(x2), int(y2)),(255,0,0),3)

            #print(f'    number of supporting lines = {len(lines_in_current_detection)}')

            if len(lines_in_current_detection)>2: #3
                #print(lines_in_current_detection)
                window_id = find_3d_line_in_windows(lines_in_current_detection)

                if image_id==1:
                    window_counter+=1
                    window_id = window_counter
                    add_lines_to_window(window_id, lines_in_current_detection, image_id)
                elif window_id is not None:
                    print(f'    updating window {window_id}')
                    add_lines_to_window(window_id, lines_in_current_detection, image_id)
                #elif len(lines_in_current_detection)>3:
                else:
                    window_counter+=1
                    window_id = window_counter
                    add_lines_to_window(window_id, lines_in_current_detection, image_id)

                #x1, y1, x2, y2 = detection.bounding_box
                #cv2.rectangle(current_image,(int(x1), int(y1)), (int(x2), int(y2)),(255,0,0),3)
                #text = f"{window_id}"
                #text_x = int(x1)
                #text_y = int(y2) + 20  # 20 pixels below the bottom of the rectangle
                #cv2.putText(current_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 1)

        #current_image = cv2.resize(current_image, (1600, 900))
        #cv2.imshow('image', current_image)
        #cv2.moveWindow('image', 2001, 100)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    #print('-----------')
    #for window_id in windows:
    #    print(windows[window_id]['3d_lines'])
  
    #print('-----------')
    #for track_id in line_to_window_map:
    #    print(f'track {track_id} for window {line_to_window_map[track_id]}')

    #all_points = []
    #for track_id in range(1, num_tracks):
    #    line = line_tracker.get_3d_line(track_id)
    #    start_point, end_point = line
    #    sampled_points = sample_points_on_line(np.array(start_point), np.array(end_point), 10)
    #    all_points.extend(sampled_points)
    #pcd = o3d.geometry.PointCloud()
    #pcd.points = o3d.utility.Vector3dVector(all_points)
    #pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=10))
    ##pcd.orient_normals_towards_camera_location(camera_location=np.array([0, 0, 0]))
    #scale_factor = .01  # Adjust this factor to increase or decrease the length of the normals
    #pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals) * scale_factor)
    #o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    #return

    # Initialize visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    num_samples_per_line = 20
    for window_id, window_data in windows.items():
        center, normal, lines = compute_normals(window_id, line_tracker)  # Ensure this function returns correct values

        # Visualize the normals
        #arrow = create_arrow(center, normal)
        #vis.add_geometry(arrow)

        # Visualize the lines
        line_set = create_line_set(lines, np.random.rand(3))  # Blue color for lines
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
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=10))
        #pcd.orient_normals_consistent_tangent_plane(k=10)
        ##pcd.orient_normals_towards_camera_location(camera_location=np.array([0, 0, 0]))
        ##o3d.visualization.draw_geometries([pcd], point_show_normal=True)

        # Visualize the point cloud with normals
        #pcd.paint_uniform_color([0, 0, 1])  # Blue color for the point cloud
        #vis.add_geometry(pcd)

        # Calculate the mean normal
        normals = np.asarray(pcd.normals)
        average_normal = np.mean(normals, axis=0)
        average_normal /= np.linalg.norm(average_normal)  # Normalize the average normal

        # Compute the geometric center of the point cloud
        points = np.asarray(pcd.points)
        center_location = np.mean(points, axis=0)

        # Create and visualize the average normal as an arrow
        arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.01, cone_radius=0.02, cylinder_height=0.1, cone_height=0.05)
        transform = np.eye(4)
        transform[:3, 3] = center_location
        transform[:3, :3] = rotation_matrix_from_vectors(np.array([0, 0, 1]), normal)
        arrow.transform(transform)
        arrow.paint_uniform_color([1, 0, 0])  # Red color for the normal
        vis.add_geometry(arrow)

        # Visualize normals as arrows
        # Assuming pcd is your point cloud
        #normals = np.asarray(pcd.normals)
        #points = np.asarray(pcd.points)
        #arrows = [o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.001, cone_radius=0.005, cylinder_height=0.05, cone_height=0.025) for _ in range(len(points))]

        #for arrow, point, normal in zip(arrows, points, normals):
        #    # Create rotation matrix from z-axis to normal vector
        #    R = rotation_matrix_from_vectors(np.array([0, 0, 1]), normal)
        #    transform = np.eye(4)
        #    transform[:3, :3] = R
        #    transform[:3, 3] = point  # Set the translation to the point location
        #    arrow.transform(transform)
        #    arrow.paint_uniform_color([1, 0, 0])  # Color the arrow red
        #    vis.add_geometry(arrow)
        
        # Create and add the plane mesh
        #mesh = create_plane_mesh(normal, center, 0.1)  # Adjust scale as needed
        #mesh.paint_uniform_color([0.5, 0.5, 0.5])  # Grey color for the plane
        #mesh.compute_vertex_normals()
        #mesh.orient_triangles()
        #additional_triangles = [tri[::-1] for tri in mesh.triangles]
        #mesh.triangles = o3d.utility.Vector3iVector(list(mesh.triangles) + additional_triangles)
        #vis.add_geometry(mesh)
      
    # Run the visualizer
    vis.run()
    vis.destroy_window()

        #for prob, bbox in detected_windows:
        #    x_min, y_min, x_max, y_max = bbox
        #    confidence = float(np.max(prob))
        #    class_label = str(trt_inference.CLASSES[np.argmax(prob)])
        #    print(f"{class_label}, {confidence}, {x_min}, {y_min}, {x_max}, {y_max}")
    
        # Track and assign IDs to windows across images
        #tracked_windows = window_tracker.track_and_assign_ids(detected_windows, P) # TODO add pose

        ## Estimate depth
        #depth_map = estimate_depth(image) # TODO link to depth estimator

        #for window in tracked_windows:
        #    # Estimate initial normals for each window
        #    initial_normals = estimate_normals_from_depth(depth_map, window) # TODO

        #    # Initialize list for this window ID if not already present
        #    if window.id not in all_estimated_normals:
        #        all_estimated_normals[window.id] = []

        #    # Store initial normals
        #    all_estimated_normals[window.id].append(initial_normals)

            ## If using multiple images for the same scene ( sliding window, maybe keyframe based optimsisation? )
            #if multiple_images_available:
            #    # 3D Reconstruction
            #    model_3D = reconstruct_3D_scene(multiple_images)
            #    projected_window = project_to_3D(window, model_3D)
            #    # Refine the normals using 3D data
            #    refined_normal = refine_normal_estimate(projected_window, model_3D, initial_normals)
            #    all_estimated_normals[window.id][-1] = refined_normal  # Update with refined normal

    ## Aggregate and refine results across all images
    #final_normals = {}
    #for window_id, normals_list in all_estimated_normals.items():
    #    # Aggregate and refine normals for each window across all images
    #    final_normals[window_id] = aggregate_and_refine_normals(normals_list)

    # 'final_normals' now contains the most accurate normal estimates for each tracked window


if __name__ == '__main__':
    args = Args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
