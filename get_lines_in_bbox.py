import os, sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# conda activate detr
# cd /home/mrt/dev/limap/
# python setup.py build
# python setup.py install
sys.path.append("../limap")
import limap.base as _base
import limap.util.io as limapio
import limap.visualize as limapvis

sys.path.append("../detr")
from detect_trt import TensorRTInference

from window_tracker import Window, WindowTracker

class Args:
    def __init__(self):
        self.data_dir = '/home/mrt/dev/windows_normal/data/dunster/'
        self.output_dir = '/home/mrt/dev/windows_normal/data/dunster/outputs'

def is_point_in_box(point, box):
    """Check if the point (x, y) is inside the bounding box defined by [x_min, y_min, x_max, y_max]."""
    x, y = point
    x_min, y_min, x_max, y_max = box
    return x_min <= x <= x_max and y_min <= y <= y_max

def do_lines_intersect(line1, box):
    """Check if a line segment (defined by two points) intersects with a box."""
    print(line1)
    (x1, y1), (x2, y2) = line1
    line_box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    # Quick bounding box check to rule out no intersection possibility
    if (line_box[2] < box[0] or line_box[0] > box[2] or
            line_box[3] < box[1] or line_box[1] > box[3]):
        return False
    
    # Line clipping as per Cohen-Sutherland or Liang-Barsky can be implemented here for precise calculation
    
    # This example assumes intersection for simplicity if bounding boxes of the line and box overlap
    return True  # Replace with detailed algorithm or library function for actual geometric computation

def main(args):
    # images
    imagecols = _base.ImageCollection(limapio.read_npy("/home/mrt/dev/limap/data/DJI/finaltracks/imagecols.npy").item())
    print(f"NumCameras = {imagecols.NumCameras()}")
    print(f"NumImages = {imagecols.NumImages()}")
    for img_id in range(1, imagecols.NumCameras()+1):
        if imagecols.exist_image(img_id):
            image = imagecols.image_name(img_id)
            #print(f"camimage = {imagecols.camimage(img_id)}")
            #print(f"campose = {imagecols.campose(img_id)}")

    # lines and detections
    lines, linetracks = limapio.read_lines_from_input("/home/mrt/dev/limap/data/DJI/finaltracks/")
    # limap.base.LineTrack: Associated line track across multi-view.
    print(f"Lines = {len(lines)}")
    print(f"LineTracks = {len(linetracks)}")

#for i in range(len(linetracks)):
#    print(f"Image Ids {linetracks[i].image_id_list}")
#    print(f"    line_id_list {linetracks[i].line_id_list}")
#    for k in range(len(linetracks[i].line2d_list)): # list[Line2d], the supporting 2D line segments
#        print(f"    {linetracks[i].line2d_list[k].coords()}")

#line = lines[0]
#print(lines)
#print(dir(line))
#print(line.length())





lines = [((10, 10), (100, 100)), ((30, 30), (60, 90)), ((40, 100), (80, 200))]
boxes = [(20, 20, 70, 70), (0, 0, 50, 50)]

# Check each line against each box
for line in lines:
    for box in boxes:
        if (is_point_in_box(line[0], box) or
                is_point_in_box(line[1], box) or
                do_lines_intersect(line, box)):
            print(f"Line {line} interacts with box {box}")

fig, ax = plt.subplots()
for line in lines:
    ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], 'blue')  # Lines in blue

for box in boxes:
    rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)  # Boxes in red

plt.show()