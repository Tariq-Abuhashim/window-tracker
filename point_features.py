import cv2
import numpy as np

def extract_features(image):
    # Create ORB detector
    orb = cv2.ORB_create()

    # Detect ORB features in the image
    keypoints, descriptors = orb.detectAndCompute(image, None)

    # Convert descriptors list to numpy array
    descriptors = np.array(descriptors)

    return keypoints, descriptors

def match_features(descriptors1, descriptors2):
    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by score
    matches = sorted(matches, key=lambda x: x.distance)

    return matches

# Example usage
image1 = cv2.imread('/home/mrt/dev/detr/data/ms/window/all_images/20230210T081219.356910.png', 0)  # Load the image in grayscale
image2 = cv2.imread('/home/mrt/dev/detr/data/ms/window/all_images/20230210T081219.678661.png', 0)

keypoints1, descriptors1 = extract_features(image1)
keypoints2, descriptors2 = extract_features(image2)

matches = match_features(descriptors1, descriptors2)

# Draw matches
#img_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#cv2.imshow('Matches', img_matches)
#cv2.waitKey()

# Assuming keypoints1, keypoints2 are lists of keypoints,
# and matches is a list of DMatch objects:
pts1 = np.float32([ keypoints1[m.queryIdx].pt for m in matches ]).reshape(-1,2)
pts2 = np.float32([ keypoints2[m.trainIdx].pt for m in matches ]).reshape(-1,2)

# Compute the fundamental matrix using RANSAC
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

# Select only inlier matches
pts1_inliers = pts1[mask.ravel() == 1]
pts2_inliers = pts2[mask.ravel() == 1]

# Draw inlier matches
# Assuming 'matches' is the original list of DMatch objects between keypoints1 and keypoints2
# and 'mask' is a binary mask where value 1 indicates an inlier match
inlier_matches = [m for m, keep in zip(matches, mask.ravel().tolist()) if keep]
img_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, inlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow('Matches', img_matches)
cv2.waitKey()


