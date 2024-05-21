import os
import numpy as np
import cv2
from scipy.linalg import null_space
import lmatch  # Placeholder for line matching functions

'''
lines: This is a list of dictionaries, where each dictionary represents a line segment. The line 
       segments are loaded from a file and then transformed. Each dictionary has two keys: 'u' and 
       'v', each corresponding to a point in 2D space (possibly the endpoints of the line segment). 
       These points are represented as 2D vectors (or arrays) with two elements each. This structure 
       is typical in computational geometry and computer vision for representing line segments in an 
       image or a 2D plane.

P: a series of arrays. Each array appears to be a 2D matrix with 3 rows and 4 columns. This suggests 
   that P is a list (or an array) of 2D matrices. In many programming languages, this could be 
   represented as a list of arrays or a multi-dimensional array. This structure is typical in 
   computer vision for representing transformation matrices or camera projection matrices.

D: is a 2D array (or matrix) with dimensions 6x6, as seen from the values provided. It's likely a 
   distance matrix, given its symmetric nature and the context of zero values along the diagonal. 
   This suggests that D[i][j] represents some form of distance or metric between elements i and j.

kk: is a 1D array (or vector) with elements [0, 3, 5]. This is a simple linear data structure, 
    likely used for indexing or referencing specific elements in another data structure, such as 
    selecting specific matrices from P or specific rows/columns in D.

opt: is a dictionary, as evident from its key-value structure. Dictionaries are data structures that 
     store pairs of elements, with each key mapping to a value. This is commonly used in programming 
     to store configurations or parameters, as each key can be a parameter name with its 
     corresponding value.
'''

# Set base directory and file extension
base_dir = './examples/aerial/small'
ext = 'png'

# Load data
names = sorted([f for f in os.listdir(base_dir) if f.endswith('.' + ext)])
images = []
camera_centers = []
lines = []
D = []
P = []

# Process each image
for name in names:
    name = os.path.splitext(name)[0]
    image = cv2.imread(os.path.join(base_dir, name + '.' + ext), cv2.IMREAD_GRAYSCALE)

    images.append(image)
    imsize = image.shape

    # Load camera parameters and reshape to a 3x4 matrix
    P_matrix = np.loadtxt(os.path.join(base_dir, name + '.P')).reshape(3, 4)
    P.append(P_matrix)
    # Compute the null space (camera center in homogeneous coordinates)
    center = null_space(P_matrix)
    # The null_space function returns a 2D array, but we expect a single vector
    # We take the first (and only) column and flatten it to get a 1D array
    center = center[:, 0]
    camera_centers.append(center)

    # Line segment detection and orientation
    lines_name = os.path.join(base_dir, f'{name}.lines')
    if not os.path.exists(lines_name):
        print(f'\nCOMPUTING LINE SEGMENTS IN IMAGE {name}')
        detected_lines = lmatch.detect_lines(image, 20)  # Implement this function
        np.savetxt(lines_name, detected_lines, fmt='%i')
        print(f' ... {len(detected_lines)} lines detected\n')
    
    # Load line segments
    loaded_lines = np.loadtxt(lines_name).T
    formatted_lines = []
    for i in range(loaded_lines.shape[1]):
        line = {
            'u': loaded_lines[:2, i],
            'v': loaded_lines[2:, i]
        }
        formatted_lines.append(line)
    oriented_lines = lmatch.orient_lines(formatted_lines, image)  # Implement this function
    lines.append(oriented_lines)

# Compute table D of pairwise view distances
for c1 in camera_centers:
    row = []
    for c2 in camera_centers:
        distance = np.linalg.norm((c1[:3] / c1[3]) - (c2[:3] / c2[3]))
        row.append(distance)
    D.append(row)
D = np.array(D)

# Set options for line matching
opt = lmatch.options()
opt['Calibration'] = 3
opt['disp_range'] = 100
opt['ordering'] = 1
opt['merging'] = 1
print(opt)

# Generate and save tentative matches
M_file = os.path.join(base_dir, 'M.mat')
M = None
if os.path.exists(M_file):
    M = np.load(M_file, allow_pickle=True)  # Load M if it exists
else:
    #for kk in lmatch.find_basepairs(D):  # Implement this function
    #    M = lmatch.generate(M, lines, P, images, D, kk, opt)  # Implement this function
    np.save(M_file, M)  # Save the generated matches
