from rtree import index #pip install rtree
import numpy as np

def subinv(P):
    """
    Compute the inverse of a 3x4 transformation matrix, assuming it represents
    a rigid-body transformation (rotation and translation).

    Args:
    P: A 3x4 transformation matrix.

    Returns:
    The 4x4 inverse transformation matrix.
    """
    # Convert P to a 4x4 matrix by adding a row [0, 0, 0, 1]
    P_extended = np.vstack([P, np.array([0, 0, 0, 1])])

    # The rotation part, R, is simply the transpose of the upper left 3x3 submatrix of P
    R = P_extended[:3, :3].T
    # The translation part, t, can be inverted by applying the inverted rotation to it
    t = -R @ P_extended[:3, 3]

    # Construct the inverse transformation matrix
    P_inv = np.eye(4)
    P_inv[:3, :3] = R
    P_inv[:3, 3] = t

    return P_inv

class Window:
    def __init__(self, bounding_box, window_id=None):
        self.bounding_box = bounding_box  # Format: (x_min, y_min, x_max, y_max)
        self.id = window_id
        #self.tracks_id = None # [list] of associated 3d line track IDs
        self.image_id = None # [list] of associated image IDs

class WindowTracker:
    def __init__(self, tracking_method="iou", intrinsic_matrix=None):
        self.tracking_method = tracking_method
        self.previous_windows = None  # Store the previous frame's windows
        self.previous_pose = None  # Store the previous frame's pose
        self.spatial_index = index.Index() # for R-tree IoU tracking
        self.next_window_id = 1  # Tracks the next available window ID
        self.intrinsic_matrix = intrinsic_matrix

    def generate_new_id(self):
        window_id = self.next_window_id
        self.next_window_id += 1
        return window_id

    def update_spatial_index(self):
        self.spatial_index = index.Index()
        for i, window in enumerate(self.previous_windows):
            self.spatial_index.insert(i, window.bounding_box)

    def track_and_assign_ids(self, current_windows, current_pose):
        if self.tracking_method == "iou":
            tracked_windows = self.track_windows_iou(current_windows, current_pose)
        elif self.tracking_method == "edges":
            tracked_windows = self.track_windows_edges(current_windows, current_pose) # TODO implement this one and check inputs and self
        else:
            raise ValueError("Invalid tracking method")

        #self.previous_windows = current_windows
        #self.previous_pose = current_pose
        return tracked_windows

    def project_bounding_box(self, box, current_pose, previous_pose):
        """
        Project the bounding box from previous pose to current pose
        This function needs to account for the intrinsic and extrinsic parameters of the camera
        and apply the necessary transformations to the bounding box coordinates
        ...
        """
        # Assuming bounding_box format is (x_min, y_min, x_max, y_max)
        x_min, y_min, x_max, y_max = box
        # Homogeneous coordinates of the bounding box corners
        corners = np.array([
            [x_min, y_min, 1],
            [x_min, y_max, 1],
            [x_max, y_min, 1],
            [x_max, y_max, 1]
        ]).T  # Shape: (3, 4)
        #print(corners)
        #print("-----------")

        # Invert the previous pose to transform from image to world space
        #prev_pose_inv = np.linalg.inv(np.vstack([previous_pose, [0, 0, 0, 1]]))[:3, :]
        prev_pose_inv = subinv(previous_pose)[:3, :]
        #print(prev_pose_inv)
        #print("-----------")

        # Transform the corners to world space
        corners_world = prev_pose_inv @ np.vstack([corners, np.ones((1, corners.shape[1]))])
        #print(corners_world)
        #print("-----------")

        # Project the world space corners to the image space using the current pose
        corners_projected = current_pose @ np.vstack([corners_world, np.ones((1, corners_world.shape[1]))])

        # Normalize the homogeneous coordinates
        corners_projected /= corners_projected[2, :]

        # Extract the new min and max x, y from the projected corners
        x_min_projected, y_min_projected = np.min(corners_projected[:2, :], axis=1)
        x_max_projected, y_max_projected = np.max(corners_projected[:2, :], axis=1)

        projected_box = (x_min_projected, y_min_projected, x_max_projected, y_max_projected)
        return projected_box

    def calculate_iou(self, box1, box2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Parameters:
        box1 (tuple): A bounding box in the format (x_min, y_min, x_max, y_max).
        box2 (tuple): A second bounding box in the format (x_min, y_min, x_max, y_max).

        Returns:
        float: the IoU of the two bounding boxes.
        """

        # Determine the coordinates of the intersection rectangle
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # Calculate area of intersection rectangle
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate areas of individual bounding boxes
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # Calculate union area
        union_area = box1_area + box2_area - intersection_area

        # Compute IoU
        iou = intersection_area / union_area

        return iou

    def detect_lines(self, image, window):
        """
        Detect lines within the window's bounding box in the given image.

        Parameters:
        image (np.array): The image in which the window is detected.
        window (Window): The window object with a bounding box.

        Returns:
        list: A list of detected lines within the window.
        """
        # Crop the image to the window's bounding box
        x_min, y_min, x_max, y_max = window.bounding_box
        window_image = image[y_min:y_max, x_min:x_max]

        # Apply edge detection
        edges = cv2.Canny(window_image, threshold1=..., threshold2=...)

        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(edges, ...)
        
        return lines

    def compare_lines(self, lines1, lines2): #TODO
        """
        Compare two sets of lines (e.g., using a similarity metric).

        Parameters:
        lines1 (list): First set of lines.
        lines2 (list): Second set of lines.

        Returns:
        float: A score representing the similarity between the two sets of lines.
        """
        # Implement a method to compare lines (e.g., based on orientation, length, distance)
        pass

    def track_windows_iou(self, current_windows, current_pose, iou_threshold=0.5):
        """
        Track windows using Intersection over Union (IoU).

        Parameters:
        current_windows (list): List of detected windows in the current image.
        previous_windows (list): List of detected windows in the previous image.
        iou_threshold (float): Threshold for IoU to consider a match.

        Returns:
        list: Tracked windows with consistent IDs across frames.
        """
        if not self.previous_windows:
            # Assign new IDs to all current windows if this is the first frame
            for window in current_windows:
                window.id = self.generate_new_id()
                print(f"----{window.id}")
            self.previous_windows = current_windows
            self.previous_pose = current_pose
            self.update_spatial_index()
            return current_windows

        tracked_windows = []
        for curr_win in current_windows:
            possible_matches = list(self.spatial_index.intersection(curr_win.bounding_box))
            best_iou = 0
            best_match_index = -1

            for idx in possible_matches:
                prev_win = self.previous_windows[idx]
                projected_box = self.project_bounding_box(prev_win.bounding_box, current_pose, self.previous_pose) # TODO validate this function
                iou = self.calculate_iou(curr_win.bounding_box, projected_box) # TODO validate this function
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_match_index = idx

            if best_match_index >= 0:
                # Assign the ID from the best matching previous window
                matched_window = self.previous_windows[best_match_index]
                curr_win.id = matched_window.id
            else:
                # Generate a new ID for new windows
                curr_win.id = self.generate_new_id()

            print(f"----{curr_win.id}")

            tracked_windows.append(curr_win)

        # Update previous windows and spatial index for the next frame
        self.previous_windows = current_windows
        self.previous_pose = current_pose
        self.update_spatial_index()

        return tracked_windows

    def track_windows_edges(self, current_image, current_windows, previous_image, previous_windows): # TODO update to look like track_windows_iou
        """
        Track windows using edges and lines detected within each window.

        Parameters:
        current_image (np.array): The current image.
        current_windows (list): List of detected windows in the current image.
        previous_image (np.array): The previous image.
        previous_windows (list): List of detected windows in the previous image.

        Returns:
        list: Tracked windows with consistent IDs across frames.
        """
        if previous_windows is None:
            return current_windows

        tracked_windows = []
        for curr_win in current_windows: # FIXME cheaper matching, maybe KD-tree with line features ?
            curr_lines = self.detect_lines(current_image, curr_win)
            best_match = None
            best_score = 0  # Define a scoring metric for line matching

            for prev_win in previous_windows:
                prev_lines = self.detect_lines(previous_image, prev_win)
                score = self.compare_lines(curr_lines, prev_lines)  # TODO Implement this method (convert Matlab code)

                if score > best_score:
                    best_score = score
                    best_match = prev_win

            if best_match:
                curr_win.id = best_match.id
            else:
                curr_win.id = generate_new_id()

            tracked_windows.append(curr_win)

        return tracked_windows

    # Additional methods as needed for tracking...