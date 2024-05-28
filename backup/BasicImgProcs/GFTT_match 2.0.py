import numpy as np
import cv2 as cv

# Load image
image1_path = "C:\\Users\\JARVIS\\Downloads\\XRP_Robot-08.jpg"
img1 = cv.imread(image1_path)

# Create a mirror image by flipping horizontally
mirror_img = cv.flip(img1, 1)  # 1: Flip horizontally, 0: Flip vertically, -1: Flip both horizontally and vertically

# Initialize the GFTT detector
gftt = cv.GFTTDetector()

# Detect keypoints in the first image
kp1, des1 = gftt.detectAndCompute(img1, None)
 
# Detect keypoints in the SECOND image
kp2, des2 = gftt.detectAndCompute(mirror_img, None)
'''
# Convert images to grayscale
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray_mirror = cv.cvtColor(mirror_img, cv.COLOR_BGR2GRAY)

# Detect good features to track
corners1 = cv.goodFeaturesToTrack(gray1, maxCorners=150, qualityLevel=0.01, minDistance=10)
corners_mirror = cv.goodFeaturesToTrack(gray_mirror, maxCorners=150, qualityLevel=0.01, minDistance=10)

# Convert detected corners to keypoints
kp1 = [cv.KeyPoint(x=c[0][0], y=c[0][1], _size=10) for c in corners1]
kp2 = [cv.KeyPoint(x=c[0][0], y=c[0][1], _size=10) for c in corners_mirror]

# Create descriptors (dummy descriptors since goodFeaturesToTrack doesn't provide descriptors)
des1 = np.zeros((len(kp1), 128), dtype=np.float32)
des2 = np.zeros((len(kp2), 128), dtype=np.float32)
'''
# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.55 * n.distance:
        good_matches.append([m])

# Select good matched keypoints
ref_matched_kpts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches])
sensed_matched_kpts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches])

# Compute homography
H, status = cv.findHomography(sensed_matched_kpts, ref_matched_kpts, cv.RANSAC, 5.0)

# Print inliers and outliers count
print("No of outliers:", len(status) - np.sum(status))
print("No of inliers:", np.sum(status))
inlier_ratio = float(np.sum(status)) / float(len(status))
print('Inlier ratio:', inlier_ratio)

# Draw matches
img3 = cv.drawMatchesKnn(img1, kp1, mirror_img, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Save the modified image to disk
cv.imwrite('C:\\Users\\JARVIS\\Documents\\Tech drive\\Dissertation\\output images\\match3.jpg', img3)  # Save the image

# Display the image with OpenCV
cv.imshow('Matched Features', img3)
cv.waitKey(0)
cv.destroyAllWindows()
