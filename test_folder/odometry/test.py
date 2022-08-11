import numpy as np 
import cv2

from VO import PinholeCamera, VisualOdometry
# from sift_odometry import PinholeCamera, VisualOdometry

cam = PinholeCamera(1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157)
vo = VisualOdometry(cam, '/vol/vssp/datasets/vid+depth/kitti/odometry/dataset/poses/00.txt')
#/vol/vssp/datasets/vid+depth/kitti/odometry/dataset

# traj = np.zeros((600,600,3), dtype=np.uint8)

# for img_id in range(4541):
# 	print(img_id)
# orb = cv2.ORB_create()




for img_id in range(4541):
	img_nxt = cv2.imread('/vol/vssp/datasets/vid+depth/kitti/odometry/dataset/sequences/00/image_0/'+str(img_id).zfill(6)+'.png', 0)
	# keyp2, disptr2 = orb.detectAndCompute(img_nxt, None)
	vo.update(img_nxt, img_id)



# img_cur = cv2.imread('/vol/vssp/datasets/vid+depth/kitti/odometry/dataset/sequences/00/image_0/'+str(0).zfill(6)+'.png', 0)
# keyp1, disptr1 = orb.detectAndCompute(img_cur, None)
#
#
# for img_id in range(1, 4541):
# 	img_nxt = cv2.imread('/vol/vssp/datasets/vid+depth/kitti/odometry/dataset/sequences/00/image_0/'+str(img_id).zfill(6)+'.png', 0)
# 	keyp2, disptr2 = orb.detectAndCompute(img_nxt, None)
# 	vo.update(img_nxt, img_id)
#
# 	# brute force match
# 	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # cC=True ==> best matches only
# 	matches = bf.match(disptr1, disptr2)
#
# 	# sorting the match vales from low 2 high
# 	matches = sorted(matches, key=lambda x: x.distance)
#
# 	# drawing the matches on the images
# 	matching_result = cv2.drawMatches(img_cur, keyp1, img_nxt, keyp2, matches[0:20], None)  # [:20]matches 0 to 20 only
#
# 	# display matches
# 	cv2.imshow("match_result", matching_result)
# 	cv2.waitKey(0)
# 	cv2.desrtroyAllWindows()
# 	img_cur = img_nxt
# 	keyp1, disptr1 = keyp2, disptr2
#
# #
# vo.update(img_nxt, img_id)




#
# 	cur_t = vo.cur_t
# 	if(img_id > 2):
# 		x, y, z = cur_t[0], cur_t[1], cur_t[2]
# 	else:
# 		x, y, z = 0., 0., 0.
# 	draw_x, draw_y = int(x)+290, int(z)+90
# 	true_x, true_y = int(vo.trueX)+290, int(vo.trueZ)+90
#
# 	cv2.circle(traj, (draw_x,draw_y), 1, (img_id*255/4540,255-img_id*255/4540,0), 1)
# 	cv2.circle(traj, (true_x,true_y), 1, (0,0,255), 2)
# 	cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
# 	text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(x,y,z)
# 	cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
#
# 	cv2.imshow('Road facing camera', img)
# 	cv2.imshow('Trajectory', traj)
# 	cv2.waitKey(1)
#
# cv2.imwrite('map.png', traj)
