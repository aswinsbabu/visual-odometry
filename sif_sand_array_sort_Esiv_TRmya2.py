#File: sif_sand_array_sort_Esiv_TRmya2.py 
#=> Parent sif_sand_array_sort_Esiv/sif_sand_ransac.py/sift_sand_array_sort.py/sift_bf_ransac

#changes:
#   Displayed R and T
#   Display inlier & OL count
#   Sort array
#   kitti: '/vol/vssp/datasets/vid+depth/kitti/odometry/dataset/sequences/00/image_0/';  /vol/research/visual_localization/experiments run/sand/SAND_features/images/smallkitti
#   OOPs last frame storage

#To do:
#   Consider scale ambiquity 
#   Plot R and T
#   
#   getpose
#   Bring back main

#Done:  
#       Essential matrix (R n T)
#       Sand key points obtained
#       reshape keyP desc matrix
#       matching
#       Insert 2 images and compute
#           display all n images (done)

# ---Stdlib---
import sys
from argparse import ArgumentParser
from pathlib import Path
import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
from utils import ops
from models import Sand


#-----load images---
from os import listdir
from os.path import isfile, join

# ---Dependencies---
import torch
import matplotlib.pyplot as plt
from imageio import imread

# ---Custom---
ROOT = Path(__file__).parent  # Path to repo
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # Prepend to path so we can use these modules

DEFAULT_MODEL_NAME = 'ckpt_G10D'  # 10 channel
DEFAULT_MODEL_PATH = ROOT / 'ckpts'

parser = ArgumentParser('SAND feature extraction demo')
parser.add_argument('--model-name', default=DEFAULT_MODEL_NAME, help='Name of model to load')
parser.add_argument('--model-path', default=DEFAULT_MODEL_PATH, help='Path to directory containing models')
#parser.add_argument('--image-file', default=DEFAULT_IMAGE, help='Path to image to run inference on')

#camera matrix
filepath ='C:/Users/JARVIS/Documents/Tech drive/Dissertation/whole code SAND_features/images/smallkitti_clr/calib2.txt'
#camera_calibMatrix = open("C:\Users\JARVIS\Documents\Tech drive\Dissertation\whole code SAND_features\images\smallkitti_clr\calib2.txt", "r").readlines()
traj = np.zeros(shape=(376, 1241, 3))

#######
K2 = np.genfromtxt(filepath, dtype='float', delimiter='').reshape(3, 4)
K1 = K2[0:3, 0:3]
camera_calibMatrix=K1

#############  

#camera_calibMatrix = list(map(float , camera_calibMatrix))# change to float
# camera_calibMatrix = np.asarray(camera_calibMatrix, dtype= float)
# print('camera calib.shape',camera_calibMatrix.shape)

#image_path=ROOT/'images'
image_path='C:/Users/JARVIS/Documents/Tech drive/Dissertation/whole code SAND_features/images/smallkitti_clr/image_l'

#image_path='/vol/research/visual_localization/experiments run/sand/SAND_features/images/lena'
onlyfiles = [ f for f in listdir(image_path) if isfile(join(image_path,f)) ]
onlyfiles.sort()
images = np.empty(len(onlyfiles), dtype=object)

mono_coordinate=[]
count =0 #no of poses
R_i = np.zeros(shape=(3, 3))
t_i = np.zeros(shape=(3, 3))

# print('onlyfiles',onlyfiles)
#load all images in folder ROOT/'images'
for n in range(0, len(onlyfiles)):
  images[n] = cv.imread( join(image_path,onlyfiles[n]) )
# print('len(images)',len(images))

#
key_sand_des = []


def get_mono_coordinates():
    # multiply by the diagonal matrix to fix our vector
    # onto same coordinate axis as true values
    diag = np.array([[-1, 0, 0],
                     [0, -1, 0],
                     [0, 0, -1]])

    adj_coord = np.matmul(diag, t_i)

    return adj_coord.flatten()


def get_absolute_scale():
    '''Used to provide scale estimation for mutliplying
              translation vectors

           Returns:
               float -- Scalar value allowing for scale estimation
           '''
    pose_file_path ='C:/Users/JARVIS/Documents/Tech drive/Dissertation/whole code SAND_features/images/smallkitti_clr/poses_00.txt'
    try:
        with open(pose_file_path) as f:
            kitti_pose = f.readlines()

    except Exception as e:
        print(e)
        raise ValueError("The pose_file_path is not valid or did not lead to a txt file")

    pose = kitti_pose[count - 1].strip().split()
    x_prev = float(pose[3]) #translation
    y_prev = float(pose[7])
    z_prev = float(pose[11])

    pose = kitti_pose[count].strip().split()
    x = float(pose[3])
    y = float(pose[7])
    z = float(pose[11])

    true_vect = np.array([[x], [y], [z]])
    true_coord = true_vect
    prev_vect = np.array([[x_prev], [y_prev], [z_prev]])


    return np.linalg.norm(true_vect - prev_vect)#mag and dir


#store key point attributes
class frame_store: #last_frame_keyPoint(x_cord,y_cord,key_sand_des)
    def __init__(self, x_cord, y_cord,keypoints, key_sand_des,xy_arrayP):

        self.x_cordinate = x_cord
        self.y_cordinate = y_cord
        self.keypoints = keypoints
        self.key_sand_des = key_sand_des
        self.xy_arrayP = xy_arrayP


    # def store_key_des(desc_points,size): #function/method to store KeyDesc
    #     for r in range(1,len(images)): #iterate for all images
    #         desc[r]=desc_points
try:

####### FOR ALL IMAGES IN THE FOLDER do SAND and SIFT ###########
    for z in range(0, 5):
    #for z in range(0, len(images)):

        def get_feat_descrpt():
            args = parser.parse_args()
            device = ops.get_device()

            ckpt_file = Path(args.model_path, args.model_name).with_suffix('.pt')
            # change
            img_file = images[z]

            # Load image & convert to torch format
            img_np = imread(img_file)
            img_torch = ops.img2torch(img_np, batched=True).to(device)

            # print(f'Image size (np): {img_np.shape}')
            # print(f'Image size (torch): {img_torch.shape}')

            # Create & load model (single branch)
            model = Sand.from_ckpt(ckpt_file).to(device)
            model.eval()

            # Run inference
            with torch.no_grad():
                features_torch = model(img_torch)
            features_np = ops.torch2np(features_torch).squeeze(0)
            return features_np
        #########################

        ####################################### SIFT ##########################

        # convert to grayscale image
        gray_scale = cv.cvtColor(images[z], cv.COLOR_BGR2GRAY)

        # initialize SIFT object
        sift = cv.SIFT_create()
        # keypoints,desc= sift.detectAndCompute(images[z], None)
        keypoints = sift.detect(images[z], None)  # descriptor is from sand
        x_cord = []
        y_cord = []
        for i in range(0, len(keypoints), 1):
            x_cord.append(int(keypoints[i].pt[0]))
            y_cord.append(int(keypoints[i].pt[1]))

        ############################ Combine x and y ############################################
        x_array = np.asarray([x_cord])  # convert to array
        y_array = np.asarray([y_cord])
        #print('x_array.shape',x_array.shape)
        xy_array= np.concatenate((x_array,y_array), axis=0)
        #print('x  y _array.shape', xy_array.shape)

        xy_array = np.transpose(xy_array)
        print('xy transpose_array.shape', xy_array.shape)
        ########################################################
        # def mains():
        args = parser.parse_args()
        device = ops.get_device()

        ckpt_file = Path(args.model_path, args.model_name).with_suffix('.pt')
        # img_file = Path(args.image_file)
        img_file = images[z]

        # Load image & convert to torch format
        img_np = images[z]
        img_torch = ops.img2torch(img_np, batched=True).to(device)

        # print(f'Image size (np): {img_np.shape}')
        # print(f'Image size (torch): {img_torch.shape}')

        # Create & load model (single branch)
        model = Sand.from_ckpt(ckpt_file).to(device)
        model.eval()

        # Run inference
        with torch.no_grad():
            features_torch = model(img_torch)

        # Changes ==> xConvert features into an images we can visualize (by PCA or normalizing)x
        # features_np = ops.fmap2img(features_torch).squeeze(0) torch2 #change fmap2img to torch2np

        features_np = ops.torch2np(features_torch).squeeze(0)

        # print(f'SAND Feature size (torch): {features_torch.shape}')
        # print(f'SAND Feature size (np): {features_np.shape}')

        ######### pick SIFT features#####
        # prev_desc= key_sand_des = []
        key_sand_des = []

        # picking descriptors of SIFT keypoints
        for y, x in zip(y_cord, x_cord):  # for the pixels cordinates x and y pick SAND desc
            key_sand_des.append(features_np[y, x])
        key_sand_des = np.asarray(key_sand_des)  # converting to array
        # reshape vector
        key_sand_des = key_sand_des.reshape(-1, 10)  # length(cordinates) x 10 dimension descriptor
        # print('key_sand_des.shape', key_sand_des.shape, '\n ')

        # matching
        if z == 0:  # avoid over riding last frame values
            xy_arrayP = xy_array
            last_frame = frame_store(x_cord, y_cord, keypoints, key_sand_des, xy_arrayP=xy_array)
            # last_frame=frame_store(x_cord, y_cord, keypoints, key_sand_des)
        else:  # only match if atleast 2 frames r present
            bf = cv.BFMatcher()  # BFMatcher with default params
            matches = bf.knnMatch(last_frame.key_sand_des, key_sand_des, k=2)
            # Apply ratio test
            good = []
            for m, n in matches:

                if m.distance < 0.75 * n.distance:
                    good.append([m])

    ####### Inlier Ratio
            # # Select good matched keypoints
            # ref_matched_kpts = np.float32([last_frame.keypoints[m[0].queryIdx].pt for m in good])
            # sensed_matched_kpts = np.float32([keypoints[m[0].trainIdx].pt for m in good])
            # # Compute homography: RANSAC
            # H, status = cv.findHomography(sensed_matched_kpts, ref_matched_kpts, cv.RANSAC, 5.0)
            # # Print inliers
            # print("No of outlier", len(status) - np.sum(status))
            # print("No of inlier", np.sum(status))
            # inlier_ratio = float(np.sum(status)) / float(len(status))
            # print('Inlier ratio=', inlier_ratio)

            # plt.imshow(img3), plt.show()
            ####### Inlier Ratio ///////////////////////////////End

            #cv.drawMatchesKnn expects list of lists as matches.
            # cv.imshow('prev_frame', images[z - 1])
            # cv.imshow('current_frame', images[z])
            # img3 = cv.drawMatchesKnn(images[z - 1], last_frame.keypoints, images[z], keypoints, good, None,
            #                          flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            # cv.imshow("image", img3)
            # cv.waitKey(0)
            # cv.destroyAllWindows()

        ##////////////////////////////////////Ess.  Matrix   ///////////////////////#########
        # print('Essential matrix operation begins.................')
        # print('\n \n xy_array',xy_array)
        # print('last_frame.xy_arrayP',last_frame.xy_arrayP)
        # print('last_frame.xy_arrayP', last_frame.xy_arrayP)
        Point1 = last_frame.xy_arrayP
        Point2 = xy_array
        
        #make point size equal
        min_size=min(Point1.shape[0],Point2.shape[0])
        Point1 = Point1[:min_size,:]
        Point2 = Point2[:min_size, :]

        E,_=cv.findEssentialMat( Point1, Point2, camera_calibMatrix, cv.RANSAC, 0.999)
        _, R, t, mask = cv.recoverPose(E, Point2,  Point1, camera_calibMatrix)

        # print('type(last_frame.xy_arrayP',type(last_frame.xy_arrayP))
        # print('type(xy_array)', type(xy_array))

        absolute_scale = get_absolute_scale() #distance considering scale
        print('absolute_scale',absolute_scale)
        # if (absolute_scale > 0.1 and abs(t[2][0]) > abs(t[0][0]) and abs(t[2][0]) > abs(t[1][0])):
        if (absolute_scale > 0.1):
            t_i = t_i + absolute_scale * R_i.dot(t)
            R_i = R.dot(R_i) #matrix mul
            print('t_i', t_i)
            print('r_i', R_i)


        diag = np.array([[-1, 0, 0],
                         [0, -1, 0],
                         [0, 0, -1]])

        adj_coord = np.matmul(diag, t_i)
        final_pose=adj_coord.flatten().tolist() #  tolist()= make a list, horizontal
        #mono_coordinate.append(final_pose)
        #print(mono_coordinate)
        #
        xy_arrayP = xy_array #comment out xy_arrayP
        last_frame = frame_store(x_cord, y_cord, keypoints, key_sand_des, xy_arrayP=xy_array)#comment out xy_arrayP
        count=count+1 #count+=1


########################################################################
except:
    import pdb
    pdb.set_trace()

#mono_coord = get_mono_coordinates()

#draw_x, draw_y, draw_z = [int(math.round(x)) for x in mono_coordinate]
#draw_x, draw_y, draw_z = [int(x) for x in mono_coordinate]

traj = cv.circle(traj, (draw_x + 400, draw_z + 100), 1, list((0, 255, 0)), 4)
cv.putText(traj, 'Estimated Odometry Position:', (30, 120), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv.imshow('trajectory', traj)
with open('mono_coord.txt', 'w') as f: #save
    for coordinate in mono_coordinate:
        f.write(f'{coordinate}\n')
# import pdb
# pdb.set_trace()


cv.imwrite("./images/trajectory.png", traj)
cv.destroyAllWindows()

#traj = np.zeros(shape=(600, 800, 3))
