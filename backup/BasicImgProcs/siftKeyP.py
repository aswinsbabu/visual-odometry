import cv2 as cv

#load image
image = cv.imread("lena.jpg")

#convert to grayscale image
gray_scale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

#initialize SIFT object
sift = cv.xfeatures2d.SIFT_create()

#detect keypoints
keypoints, _= sift.detectAndCompute(image, None)

#draw keypoints
sift_image = cv.drawKeypoints(gray_scale, keypoints, None)

#show image
cv.imshow("Features Image", sift_image)

#show kwy points
#print("keypoints/t",keypoints,"/n")
print("keypoints.shape =/t",keypoints.shape,"/n")

#hold the window
cv.waitKey(0)
