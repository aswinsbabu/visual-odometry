import cv2 as cv
import numpy as np
#load image
image = cv.imread("lena.jpg")

#convert to grayscale image
gray_scale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

#initialize SIFT object
sift = cv.xfeatures2d.SIFT_create()

#detect keypoints

keypoints,abc= sift.detectAndCompute(image, None)
print("keypoints",keypoints,len(abc))

#pts = np.float([keypoints[idx].pt for idx in len(keypoints)]).reshape(-1, 1, 2)

'''
#draw keypoints
sift_image = cv.drawKeypoints(gray_scale, keypoints, None)

#show image
cv.imshow("Features Image", sift_image)

#hold the window
cv.waitKey(0)
'''
######## ##########
(h,w)=gray_scale.shape
######## ##########

######## print cordinates of keypoints ##########
for i in keypoints:
	print(i.pt)
######## ##########
'''
######## ##########
for x in range of h:
	for x in range of h:
		gray_scale[x,y]
###################	
'''
