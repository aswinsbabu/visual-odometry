
#Flip image 
#Use imshow instead of matplot
import numpy as np
import cv2 as cv

# Load image
#image1_path = "C:\\Users\\JARVIS\\Downloads\\DigiKey.jpg"# Load image
#image1_path = "C:\\Users\\JARVIS\\Downloads\\photo.jpg"
#image1_path = "C:\\Users\\JARVIS\\Downloads\\resistrclrcode.jpg"
image1_path = "C:\\Users\\JARVIS\\Downloads\\cross_ref.jpg"
#image1_path ="C:\\Users\\JARVIS\\Downloads\\XRP_Robot-08.jpg"#"C:\Users\JARVIS\Downloads\XRP_Robot-08.jpg"
#image1_path = 'C:\\Users\\JARVIS\\Documents\\Tech drive\\Dissertation\\whole code SAND_features\\images\\sample.png'

img1 = cv.imread(image1_path)

#image2_path = "C:\\Users\\JARVIS\\Downloads\\photo.jpg"
#img2 = cv.imread(image2_path)

# Create a mirror image by flipping horizontally
mirror_img = cv.flip(img1, 1) #1: Flip horizont 0: Vertical -1: Flip both H and V

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

# Draw matches
#img3 = cv.drawMatchesKnn(img1, kp1, mirror_img, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# Draw matches with increased line width
img3 = cv.drawMatchesKnn(img1, kp1, mirror_img, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), matchesMask=None)
# Save the modified image to disk
#cv.imwrite('C:\\Users\\JARVIS\\Documents\\Tech drive\\Dissertation\\output images\\DigiKEYmatch.jpg', img3)  # Save the image

# Display the image with OpenCV
cv.imshow('Matched Features', img3)
cv.waitKey(0)
cv.destroyAllWindows()
