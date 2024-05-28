import cv2 as cv
import numpy as np

# Load images
#img1 = cv2.imread("C:\\Users\\JARVIS\\Documents\\Tech drive\\Dissertation\\2024_new\\InputImg\\subset\\cross_ref.jpg", cv2.IMREAD_GRAYSCALE)
img1 = cv.imread("C:\\Users\\JARVIS\\Documents\\Tech drive\\Dissertation\\2024_new\\InputImg\\XRP_Robot.jpg", cv.IMREAD_GRAYSCALE)

# Create a mirror image by flipping horizontally
img2 = cv.flip(img1, 1)  # 1: Flip horizontally, 0: Vertical, -1: Flip both H and V

# Check if the images have been loaded correctly
if img1 is None or img2 is None:
    print("Error loading images")
    exit(1)

# Initialize ORB detector
orb = cv.ORB_create()

# Find keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# BFMatcher with default params
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Sort matches by distance
matches = sorted(matches, key = lambda x:x.distance)

# Draw lines connecting matching keypoints
img_matches = np.hstack((img1, img2))  # Create a combined image by stacking the two images side-by-side
for match in matches[:10]:  # Draw only the first 10 matches for clarity
    pt1 = tuple(map(int, kp1[match.queryIdx].pt))  # Keypoint in the first image
    pt2 = tuple(map(int, (kp2[match.trainIdx].pt[0] + img1.shape[1], kp2[match.trainIdx].pt[1])))  # Keypoint in the second image (offset by width of img1)
    cv.line(img_matches, pt1, pt2, (50, 250, 20), 2)  # Draw a line connecting the points

# Display the image with matching lines
cv.imshow('Matched Keypoints', img_matches)
cv.waitKey(0)
cv.destroyAllWindows()
