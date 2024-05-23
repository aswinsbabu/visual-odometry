import numpy as np
import cv2 as cv

# Load image
image1_path = "C:\\Users\\JARVIS\\Downloads\\XRP_Robot-08.jpg"
img1 = cv.imread(image1_path)

# Create a mirror image by flipping horizontally
mirror_img = cv.flip(img1, 1)

# Initiate SIFT detector
sift = cv.SIFT_create()

# Find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(mirror_img, None)

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

# Create an image with a black background
height, width = img1.shape[:2]
img3 = np.zeros((height, width*2, 3), dtype=np.uint8)

# Draw matches with increased line width
for match in good_matches:
    pt1 = (int(kp1[match[0].queryIdx].pt[0]), int(kp1[match[0].queryIdx].pt[1]))
    pt2 = (int(kp2[match[0].trainIdx].pt[0] + width), int(kp2[match[0].trainIdx].pt[1]))
    cv.line(img3, pt1, pt2, (0, 255, 0), thickness=3)

# Save the modified image to disk
cv.imwrite('C:\\Users\\JARVIS\\Documents\\Tech drive\\Dissertation\\output images\\DigiKEYmatch2.jpg', img3)  # Save the image

# Display the image with OpenCV
cv.imshow('Matched Features', img3)
cv.waitKey(0)
cv.destroyAllWindows()
