import cv2
import numpy as np

# Load images
img1 = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('lena_flipped.jpg', cv2.IMREAD_GRAYSCALE)

# Check if the images have been loaded correctly
if img1 is None or img2 is None:
    print("Error loading images")
    exit(1)

# Initialize the ORB detector
orb = cv2.ORB_create()

# Detect keypoints and compute descriptors for both images
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Convert descriptors to float32 (required by FLANN)
des1 = np.float32(des1)
des2 = np.float32(des2)

# FLANN parameters
FLANN_INDEX_LSH = 6
index_params = dict(algorithm = FLANN_INDEX_LSH,
                    table_number = 6, # 12
                    key_size = 12,     # 20
                    multi_probe_level = 1) #2
search_params = dict(checks=50)   # or pass empty dictionary

# Initialize the FLANN based matcher
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Perform the matching between the descriptors
matches = flann.knnMatch(des1, des2, k=2)

# Apply ratio test to select good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Draw matches
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the matches
cv2.imshow('ORB + FLANN Feature Matching', img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()
