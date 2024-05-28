import cv2
import numpy as np

# Load images
img1 = cv2.imread('C:\\Users\\JARVIS\\Documents\\Tech drive\\Dissertation\\2024_new\\lena.jpg', cv2.IMREAD_GRAYSCALE)

 # Create a mirror image by flipping horizontally
img2 = cv2.flip(img1, 1)  # 1: Flip horizontally, 0: Vertical, -1: Flip both H and V

#img2 = cv2.imread('lena_flipped.jpg', cv2.IMREAD_GRAYSCALE)

# Check if the images have been loaded correctly
if img1 is None or img2 is None:
    print("Error loading images")
    exit(1)

# Initialize the ORB detector
orb = cv2.ORB_create()

# Detect keypoints and compute descriptors for both images
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Check if descriptors are None (could be due to no keypoints found)
if des1 is None or des2 is None:
    print("Error: No descriptors found")
    exit(1)

# Convert descriptors to float32
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
    if m.distance < 0.55 * n.distance:
        good_matches.append(m)

# Draw matches with default settings
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Draw thicker lines for the matches
for match in good_matches:
    pt1 = tuple(map(int, kp1[match.queryIdx].pt))
    pt2 = tuple(map(int, (kp2[match.trainIdx].pt[0] + img1.shape[1], kp2[match.trainIdx].pt[1])))  # Offset x by width of img1
    cv2.line(img_matches, pt1, pt2, (0, 255, 0), 5)  # Change 2 to any thickness you want

# Display the matches
cv2.imshow('ORB + FLANN Feature Matching', img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()
