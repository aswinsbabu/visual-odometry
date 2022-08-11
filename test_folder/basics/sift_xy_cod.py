import cv2 as cv
import os
import numpy
#load image
image = cv.imread("lena.jpg")
l=0
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

#print coordinates
l=len(keypoints)
#i=0
for i in range(0,len(keypoints),1):
 print(i,'keypoints_cordinate',keypoints[i].pt)


# for i in range(0,1152,1):
#  print('keypoints_cordinate',keypoints[i].pt)

#print('keypoints',keypoints[].pt)

#draw keypoints
#sift_image = cv.drawKeypoints(gray_scale, keypoints, None)

#show image
#cv.imshow("Features Image", sift_image)

#hold the window
#cv.waitKey(0)

