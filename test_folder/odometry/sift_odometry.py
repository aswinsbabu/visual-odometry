import numpy as np
import cv2

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 1500
orb = cv2.ORB_create()

lk_params = dict(winSize  = (21, 21),
				#maxLevel = 3,
             	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

############## Edit this portion ###############
#Add SIFT

def featureTracking(image_ref, image_cur, px_ref):
	# kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)  #shape: [k,2] [k,1] [k,1]

	# st = st.reshape(st.shape[0])
	
	#initialize SIFT object
	sift = cv2.xfeatures2d.SIFT_create()

	#detect keypoints
	kp1, _= sift.detectAndCompute(image_ref, None)
	kp2, _= sift.detectAndCompute(image_cur, None)
	'''
	kp1 = px_ref[st == 1]
	kp2 = kp2[st == 1]
	'''
	return kp1, kp2

'''  SIFT
import cv2 as cv

#load image
image = cv.imread("lena.jpg")

#convert to grayscale image
gray_scale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

#initialize SIFT object
sift = cv.xfeatures2d.SIFT_create()

#detect keypoints
keypoints, _= sift.detectAndCompute(image, None)
'''
#################

class PinholeCamera:
	def __init__(self, width, height, fx, fy, cx, cy,
				k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
		self.width = width
		self.height = height
		self.fx = fx
		self.fy = fy
		self.cx = cx
		self.cy = cy
		self.distortion = (abs(k1) > 0.0000001)
		self.d = [k1, k2, p1, p2, k3]


class VisualOdometry:
	def __init__(self, cam, annotations):
		self.frame_stage = 0
		self.cam = cam
		self.new_frame = None
		self.last_frame = None
		self.cur_R = None
		self.cur_t = None
		self.px_ref = None
		self.px_cur = None
		self.keyp1 = None
		self.disptr1 = None
		self.keyp2 = None
		self.disptr2 = None

		self.focal = cam.fx
		self.pp = (cam.cx, cam.cy)
		self.trueX, self.trueY, self.trueZ = 0, 0, 0
		self.detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
		with open(annotations) as f:
			self.annotations = f.readlines()

	def getAbsoluteScale(self, frame_id):  #specialized for KITTI odometry dataset
		ss = self.annotations[frame_id-1].strip().split()
		x_prev = float(ss[3])
		y_prev = float(ss[7])
		z_prev = float(ss[11])
		ss = self.annotations[frame_id].strip().split()
		x = float(ss[3])
		y = float(ss[7])
		z = float(ss[11])
		self.trueX, self.trueY, self.trueZ = x, y, z
		return np.sqrt((x - x_prev)*(x - x_prev) + (y - y_prev)*(y - y_prev) + (z - z_prev)*(z - z_prev))

	def processFirstFrame(self):
		# self.px_ref = self.detector.detect(self.new_frame)
		keyp1, disptr1 = orb.detectAndCompute(self.new_frame, None)
		self.keyp1 = np.array([x.pt for x in keyp1], dtype=np.float32)
		self.disptr1 = disptr1
		self.frame_stage = STAGE_SECOND_FRAME

	def processSecondFrame(self):
		# self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
		keyp2, disptr2 = orb.detectAndCompute(self.new_frame, None)
		self.keyp2 = np.array([x.pt for x in keyp2], dtype=np.float32)

		# brute force match
		bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # cC=True ==> best matches only
		matches = bf.match(self.disptr1, disptr2)

		# sorting the match vales from low 2 high
		matches = sorted(matches, key=lambda x: x.distance)
		matches = matches[0:20]
		queryIdx = np.array([x.queryIdx for x in matches], dtype=np.int)
		trainIdx = np.array([x.trainIdx for x in matches], dtype=np.int)

		self.keyp1 = self.keyp1[queryIdx]
		self.keyp2 = self.keyp2[trainIdx]
		# matching_result = cv2.drawMatches(self., keyp1, img2, keyp2, matches[0:20], None)  # [:20]matches 0 to 20 only

		E, mask = cv2.findEssentialMat(self.keyp2, self.keyp1, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
		_, self.cur_R, self.cur_t, mask = cv2.recoverPose(E, self.keyp2, self.keyp1, focal=self.focal, pp = self.pp)


		#
		# # drawing the matches on the images
		# matching_result = cv2.drawMatches(img_cur, self.keyp1, img_nxt, keyp2, matches[0:20],
		# 								  None)  # [:20]matches 0 to 20 only
		#
		# # display matches
		# cv2.imshow("match_result", matching_result)
		# cv2.waitKey(0)
		# cv2.desrtroyAllWindows()
		# img_cur = img_nxt
		# keyp1, disptr1 = keyp2, disptr2
		#
		self.frame_stage = STAGE_DEFAULT_FRAME
		self.keyp1 = self.keyp2

	def processFrame(self, frame_id):
		self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
		E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
		_, R, t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)
		absolute_scale = self.getAbsoluteScale(frame_id)
		if(absolute_scale > 0.1):
			self.cur_t = self.cur_t + absolute_scale*self.cur_R.dot(t)
			self.cur_R = R.dot(self.cur_R)
		if(self.px_ref.shape[0] < kMinNumFeature):
			self.px_cur = self.detector.detect(self.new_frame)
			self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)
		self.px_ref = self.px_cur

	def update(self, img, frame_id):
		assert(img.ndim==2 and img.shape[0]==self.cam.height and img.shape[1]==self.cam.width), "Frame: provided image has not the same size as the camera model or image is not grayscale"
		self.new_frame = img
		if(self.frame_stage == STAGE_DEFAULT_FRAME):
			self.processFrame(frame_id)
		elif(self.frame_stage == STAGE_SECOND_FRAME):
			self.processSecondFrame()
		elif(self.frame_stage == STAGE_FIRST_FRAME):
			self.processFirstFrame()
		self.last_frame = self.new_frame

	def update(self, img, frame_id):
		assert(img.ndim==2 and img.shape[0]==self.cam.height and img.shape[1]==self.cam.width), "Frame: provided image has not the same size as the camera model or image is not grayscale"
		self.new_frame = img
		if(self.frame_stage == STAGE_DEFAULT_FRAME):
			self.processFrame(frame_id)
		elif(self.frame_stage == STAGE_SECOND_FRAME):
			self.processSecondFrame()
		elif(self.frame_stage == STAGE_FIRST_FRAME):
			self.processFirstFrame()
		self.last_frame = self.new_frame
