import cv2 as cv
import os
#load image
image = cv.imread("lena.jpg")

#convert to grayscale image
gray_scale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

#initialize SIFT object
sift = cv.xfeatures2d.SIFT_create()

#detect keypoints
keypoints, desc= sift.detectAndCompute(image, None)
print('Key_P-length', len(keypoints))
print('Key_P-type', type(keypoints))

print('desc_length', len(desc))
print('desc_type', type(desc))
print('keypoints_cordinate',keypoints[0].pt)
#print('keypoints',keypoints[].pt)

#draw keypoints
#sift_image = cv.drawKeypoints(gray_scale, keypoints, None)

#show image
#cv.imshow("Features Image", sift_image)

#hold the window
cv.waitKey(0)

