import cv2
import numpy as np

# Load images
img1 = cv2.imread('C:\\Users\\JARVIS\\Documents\\Tech drive\\Dissertation\\2024_new\\lena.jpg', cv2.IMREAD_GRAYSCALE)
# Create a mirror image by flipping horizontally
img2 = cv2.flip(img1, 1)  # 1: Flip horizontally, 0: Vertical, -1: Flip both H and V

# cv2.imshow('ORB + FLANN Feature Matching', img1)
# cv2.waitKey(0) & 0xFF == ord('q')
# cv2.destroyAllWindows()

# Check if the images have been loaded correctly
if img1 is None or img2 is None:
    print("Error loading images")
    exit(1)

# Initialize the ORB detector
orb = cv2.ORB_create()

# Detect keypoints and compute descriptors for both images
keypoints_img1, descriptors_img1 = orb.detectAndCompute(img1, None)
keypoints_img2, descriptors_img2 = orb.detectAndCompute(img2, None)

# Check if descriptors are None (could be due to no keypoints found)
if descriptors_img1 is None or descriptors_img2 is None:
    print("Error: No descriptors found")
    exit(1)
###########################################################################
# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50) # or pass empty dictionary

# apply FLANN based matcher with knn
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(descriptors_img1,descriptors_img2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
   if m.distance < 0.1*n.distance:
      matchesMask[i]=[1,0]
draw_params = dict(matchColor = (0,255,0),singlePointColor = (255,0,0),matchesMask = matchesMask,flags = 0)
img3 = cv2.drawMatchesKnn(img1,keypoints_img1,img2,keypoints_img2,matches,None,**draw_params)
plt.imshow(img3),plt.show()

'''
# Convert descriptors to float32
descriptors_img1 = np.float32(descriptors_img1).reshape(-1, 32)  # Assuming 32-dimensional descriptors
descriptors_img2 = np.float32(descriptors_img2).reshape(-1, 32)  # Assuming 32-dimensional descriptors

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
matches = flann.knnMatch(descriptors_img1, descriptors_img2, k=5)

# Apply ratio test to select good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Draw matches
img_matches = cv2.drawMatches(img1, keypoints_img1, img2, keypoints_img2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the matches
cv2.imshow('ORB + FLANN Feature Matching', img_matches)
cv2.waitKey(0) & 0xFF == ord('q')
cv2.destroyAllWindows()
'''