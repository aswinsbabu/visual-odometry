import cv2 as cv
import numpy as np
#load image
image = cv.imread("lena.jpg")

#convert to grayscale image
gray_scale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

#initialize SIFT object
sift = cv.SIFT_create()
#sift = cv.xfeatures2d.SIFT_create()

#detect keypoints

keypoints,desc= sift.detectAndCompute(image, None)

print('len of descriptor',len(desc))
print('desc',desc)
#pts = np.float([keypoints[idx].pt for idx in len(keypoints)]).reshape(-1, 1, 2)

######## ##########
(h,w)=gray_scale.shape
######## ##########

######## print cordinates of keypoints ##########
for i in range(0,len(keypoints),1):
 print(i,'keypoints_cordinate',keypoints[i].pt) 
print(keypoints[i].octave)#octave:	pyramid octave in which the keypoint has been detectedp_cord
