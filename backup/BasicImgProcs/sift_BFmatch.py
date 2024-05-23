#File name: sif_bf_ransac.py ==> Parent sift_sif_bf
#change:
#   Display inlier & OL count
#
#
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Load image
#image1_path = "C:\\Users\\JARVIS\\Downloads\\photo.jpg"
image1_path ="C:\\Users\\JARVIS\\Downloads\\DigiKey.jpg"
#image1_path ="C:\\Users\\JARVIS\\Downloads\\XRP_Robot-08.jpg"#"C:\Users\JARVIS\Downloads\XRP_Robot-08.jpg"
#image1_path = 'C:\\Users\\JARVIS\\Documents\\Tech drive\\Dissertation\\whole code SAND_features\\images\\sample.png'

img1 = cv.imread(image1_path)
#img1 = cv.imread('lena.jpg',cv.IMREAD_GRAYSCALE)          # queryImage

image2_path = "C:\\Users\\JARVIS\\Downloads\\photo.jpg"
img2= cv.imread(image2_path)
#img2 = cv.imread('lena-horizonal-flip.jpg',cv.IMREAD_GRAYSCALE) # trainImage

# Initiate SIFT detector
sift = cv.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)

# Apply ratio test
good_matches = []
for m,n in matches:
   if m.distance < 0.75*n.distance:
       good_matches.append([m])


# Select good matched keypoints
ref_matched_kpts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches])
sensed_matched_kpts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches])
 # Compute homography
H, status = cv.findHomography(sensed_matched_kpts, ref_matched_kpts, cv.RANSAC,5.0)
# number of detected outliers: len(status) - np.sum(status)
# number of detected inliers: np.sum(status)
# Inlier Ratio, number of inlier/number of matches:
#Print inliers
print("No of outlier",len(status) - np.sum(status))
print("No of inlier",np.sum(status))
inlier_ratio=float(np.sum(status)) / float(len(status))
print('Inlier ratio=',inlier_ratio)

# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good_matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()
