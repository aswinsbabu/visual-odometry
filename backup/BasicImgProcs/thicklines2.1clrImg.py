# Matches are drawn on clr images 
# added imwrite
import cv2
import numpy as np
import os

# Load images
Clr_img1 = cv2.imread('D:\\Tech drive\\Dissertation\\2024_new\\InputImg\\cross_ref.jpg')
img1 = cv2.imread("D:\\Tech drive\\Dissertation\\2024_new\\InputImg\\cross_ref.jpg", cv2.IMREAD_GRAYSCALE)

#save image path
save_path ="D:\\Tech drive\\Dissertation\\2024_new\\output images\\matchOut\\orbFlann"

# Create a mirror image by flipping horizontally
img2 = cv2.flip(img1, 1)  # 1: Flip horizontally, 0: Vertical, -1: Flip both H and V
Clr_img2 = cv2.flip(Clr_img1, 1)  # 1: Flip horizontally, 0: Vertical, -1: Flip both H and V
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
matches = flann.knnMatch(descriptors_img1, descriptors_img2, k=50)

# Apply ratio test to select good matches
good_matches = []
for match in matches:
    if len(match) > 1 and match[0].distance < 0.8 * match[1].distance:
        good_matches.append(match[0])

# Draw matches
img_matches = cv2.drawMatches(Clr_img1, keypoints_img1, Clr_img2, keypoints_img2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

#Save image
#cv2.imwrite('C:\\Users\\JARVIS\\Documents\\output images\\gftt_xrp08.jpg)
#cv2.imwrite(output_path,'thickClr.jpg')
file_name='thickline2.0Clr'
file_extension='.jpg'
new_file_name = file_name + '_new' + file_extension
new_file_path = os.path.join(save_path, new_file_name)
cv2.imwrite(new_file_path, img_matches )  # Save the image
        
# Display the matches
cv2.imshow('ORB + FLANN Feature Matching', img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()