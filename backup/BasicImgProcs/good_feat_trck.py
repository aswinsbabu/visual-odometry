#import
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

#load image
#image_path = 'C:\\Users\\JARVIS\\Documents\\Tech drive\\Dissertation\\old experiments run\\feature_match\\lena.jpg'
#image_path = 'C:\\Users\\JARVIS\\Documents\\Tech drive\\Dissertation\\whole code SAND_features\\images\\sample.png'
#image_path = "C:\\Users\\JARVIS\\Downloads\\photo.jpg"
image_path ="C:\\Users\\JARVIS\\Downloads\\DigiKey.jpg"
#C:\Users\JARVIS\Documents\Tech drive\Dissertation\whole code SAND_features\images
img = cv.imread(image_path)
#img = cv.imread('lena.jpg')
#cv.imshow('lena',img)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)


corners = cv.goodFeaturesToTrack(gray,150,0.01,10)#150= Max no corners 2 rtn: 0.01= Qlty param. 10 = Min Euclid distance b/w crnrs
corners = np.int0(corners) #cnvrt to integer coordinates

for i in corners:
    x,y = i.ravel()
    cv.circle(img,(x,y),5,250,2)#(image name,<x & y coordinates>, radius, color, thickness of circle line(-1=filled) )
plt.imshow(img),plt.show()
