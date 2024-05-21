import cv2
import numpy as np
from lmatch import lmatch, vgg

image1 = cv2.imread('/home/mrt/dev/windows_normal/data/dunster/001.png', 0)  # Load the image in grayscale
image2 = cv2.imread('/home/mrt/dev/windows_normal/data/dunster/002.png', 0)

# Create LSD detector
lsd = cv2.createLineSegmentDetector(0)

# Detect lines in the images
lines1, _, _, _ = lsd.detect(image1)
lines2, _, _, _ = lsd.detect(image2)

# Example of drawing a line on an image
for line in lines1:
    #print(line)
    x0, y0, x1, y1 = map(int, line[0])
    cv2.line(image1, (x0, y0), (x1, y1), (255, 0, 0), 3)
cv2.imwrite("inlier_matches.jpg", image1)

#lmatch.lmatch_detect_lines(image1, 10)

# Edge detection using Canny
edges = cv2.Canny(image1, threshold1=20, threshold2=100, apertureSize=3, L2gradient=True)
    
# Find contours (edgel strips) from Canny edges
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
# Convert contours to the expected format for vgg_linesegs_from_edgestrips
e = [contour[:, 0, ::-1].T for contour in contours]  # Reverse X and Y to match MATLAB format
    
# Detect line segments from edgel strips
u, v, _ = vgg.vgg_linesegs_from_edgestrips(e)  # Assuming implementation is provided
    
# Filter lines based on length
minLength = 20
lengths = np.sum((u - v) ** 2, axis=0)
indices = np.where(lengths >= minLength**2)[0]
print(u)
print(v)
u = u[:, indices]
v = v[:, indices]