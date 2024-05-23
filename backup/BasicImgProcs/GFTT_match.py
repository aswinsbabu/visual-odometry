import numpy as np
import cv2 as cv

# Load image
image1_path = "C:\\Users\\JARVIS\\Downloads\\XRP_Robot-08.jpg"
img1 = cv.imread(image1_path)

# Create a mirror image by flipping horizontally
mirror_img = cv.flip(img1, 1) # 1: Flip horizontally

# Convert both images to grayscale
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(mirror_img, cv.COLOR_BGR2GRAY)

# Use goodFeaturesToTrack to find keypoints
corners1 = cv.goodFeaturesToTrack(gray1, maxCorners=500, qualityLevel=0.01, minDistance=10)
corners2 = cv.goodFeaturesToTrack(gray2, maxCorners=500, qualityLevel=0.01, minDistance=10)

# Convert corners to keypoint format
kp1 = [cv.KeyPoint(x=float(c[0][0]), y=float(c[0][1]), _size=10) for c in corners1]
kp2 = [cv.KeyPoint(x=float(c[0][0]), y=float(c[0][1]), _size=10) for c in corners2]

# Compute descriptors using ORB (since goodFeaturesToTrack does not provide descriptors)
orb = cv.ORB_create()
kp1, des1 = orb.compute(gray1, kp1)
kp2, des2 = orb.compute(gray2, kp2)

# BFMatcher with default params
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Sort matches by distance (best matches first)
matches = sorted(matches, key=lambda x: x.distance)

# Apply ratio test manually, if needed
good_matches = matches[:50]  # Select top 50 matches (arbitrary choice for demonstration)

# Select good matched keypoints
ref_matched_kpts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
sensed_matched_kpts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

# Compute homography
H, status = cv.findHomography(sensed_matched_kpts, ref_matched_kpts, cv.RANSAC, 5.0)

# Print inliers and outliers count
num_inliers = np.sum(status)
num_outliers = len(status) - num_inliers
print("No of outliers:", num_outliers)
print("No of inliers:", num_inliers)
inlier_ratio = float(num_inliers) / float(len(status))
print('Inlier ratio:', inlier_ratio)

# Draw matches
img3 = cv.drawMatches(img1, kp1, mirror_img, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Save the modified image to disk
cv.imwrite('C:\\Users\\JARVIS\\Documents\\Tech drive\\Dissertation\\output images\\match3.jpg', img3)  # Save the image

# Display the image with OpenCV
cv.imshow('Matched Features', img3)
cv.waitKey(0)
cv.destroyAllWindows()
