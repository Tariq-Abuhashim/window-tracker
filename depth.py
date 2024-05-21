import torch
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import numpy as np
import open3d as o3d


# Load a pre-trained model (example model - replace with actual model path or URL)
model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
model.eval()

# Define image transforms
transform = Compose([ToTensor(), Resize(800), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Load and preprocess an image
image_path = '/media/mrt/Whale/Orin/annotate/images/train/20230210T081220.663639.png'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
input_tensor = transform(image_rgb).unsqueeze(0)

# Perform depth estimation
with torch.no_grad():
    depth = model(input_tensor)
depth_map = depth.squeeze().numpy()

height, width = depth_map.shape

# Calculate gradients
#dx = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
#dy = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
# Assume depth is in some unit, and we set the perpendicular vector component (-1 for simplicity)
#dz = -1 * np.ones_like(dx)
# Normalize the vectors (divide by the magnitude to get unit length vectors)
#magnitude = np.sqrt(dx**2 + dy**2 + dz**2)
#normals = np.dstack((dx, dy, dz)) / magnitude[:,:,np.newaxis]
# Convert to a normal map with values between 0 and 255 (for visualization)
#normal_map = ((normals + 1) * 0.5 * 255).astype(np.uint8)

# Assuming depth_map is your depth image
gy, gx = np.gradient(depth_map)
normals = np.dstack((-gx, -gy, np.ones_like(depth_map)))
n = np.linalg.norm(normals, axis=2)
normals /= n[..., np.newaxis]

# Normalize and scale for visualization
normals_vis = ((normals + 1) / 2 * 255).astype(np.uint8)


# The normal map can now be visualized or saved
#cv2.imshow("Normal Map", normal_map)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


# Visualize the result
#plt.imshow(depth_image, cmap='inferno')
#plt.show()


# Camera intrinsic parameters
f_x = 950 # Focal length in x
f_y = 950 # Focal length in y
c_x = width / 2  # Assuming center of the image as optical center
c_y = height / 2

# Create a meshgrid of pixel coordinates
u, v = np.meshgrid(np.arange(width), np.arange(height))

# Convert to 3D coordinates
Z = depth_image.astype(float)  # Depth
X = (u - c_x) * Z / f_x
Y = (v - c_y) * Z / f_y

# Combine and reshape to get a list of 3D points [N, 3]
points_3D = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

# Filter out points with invalid depth (e.g., depth == 0 or some threshold)
valid_points = points_3D[(Z.ravel() > 20)]




# Convert numpy array to Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(valid_points)

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])
