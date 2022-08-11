#Feature detection using ORB

#import cv2 library
import cv2
import numpy as np

class detector:
	#ORB initialisation
	orb = cv2.ORB_create()
	
	def detectAndMatch(image1, image2):
		#load image
		img1= cv2.imread(image1,cv2.IMREAD_GRAYSCALE)
		img2= cv2.imread(image2, cv2.IMREAD_GRAYSCALE)

		#detect key points and descriptors
		keyp1, disptr1 = orb.detectAndCompute(img1,None)
		keyp2, disptr2 = orb.detectAndCompute(img2,None)

		#brute force match
		bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # cC=True ==> best matches only
		matches = bf.match(disptr1, disptr2)

		########### visualisation part ###################

		#sorting the match vales from low 2 high
		matches = sorted(matches, key = lambda x:x.distance)

		#drawing the matches on the images
		matching_result = cv2.drawMatches(img1,keyp1, img2,keyp2, matches[0:20], None) #[:20]matches 0 to 20 only 	
			
		#display matches
		cv2.imshow("match_result", matching_result)
		#cv2.imshow("keypoint_image2", keyp_image2)
		cv2.waitKey(0)
		cv2.desrtroyAllWindows()

