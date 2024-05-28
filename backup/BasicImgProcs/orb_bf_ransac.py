#File name: orb_bf_ransac.py ==> Parent sift_sif_bf
#change:
#   Display inlier & OL count(Working with default parameters)
#   if m.distance < 0.9*n.distance:  value changed
#
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img1 = cv.imread('lena.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('lena-horizonal-flip.jpg',cv.IMREAD_GRAYSCALE) # trainImage

#ORB initialisation
orb = cv.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

############### BFMatcher with default params####////////////////////////////////

bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)

# Apply ratio test
good_matches = []
for m,n in matches:
    print('m,n=',m, n)
    if m.distance < 0.8*n.distance:
       good_matches.append([m])
########################//////////////////////

###################NORM_HAMMING, #####-------------------------################
# create BFMatcher object
# bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# # Match descriptors.
# matches = bf.match(des1,des2)
# # Sort them in the order of their distance.
# good_matches = sorted(matches, key = lambda x:x.distance)
# # Draw first 10 matches.
# img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(img3),plt.show()
########################-------------------------################

####################### MAtch Quality analysis#######################
# Select good matched keypoints
ref_matched_kpts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches])
sensed_matched_kpts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches])
#
# ref_matched_kpts = np.float32([kp1[m[0].queryIdx].pt for m in range(good_matches)])
# sensed_matched_kpts = np.float32([kp2[m[0].trainIdx].pt for m in range( good_matches)])

 # Compute homography
H, status = cv.findHomography(sensed_matched_kpts, ref_matched_kpts, cv.RANSAC,5.0)

# #Print inliers
# print("No of outlier",len(status) - np.sum(status))
# print("No of inlier",np.sum(status))
# inlier_ratio=float(np.sum(status)) / float(len(status))
# print('Inlier ratio=',inlier_ratio)

# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good_matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()
