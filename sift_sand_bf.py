#File name: sift_sand_bf 1.0 py ==> Parent 2.3.2/main_siv_2.3.2
#change:
#   kiiti: '/vol/vssp/datasets/vid+depth/kitti/odometry/dataset';/vol/vssp/datasets/vid+depth/kitti/odometry/dataset/sequences/00
#   OOPs last frame storage
#   matching(BF)
#To do:
#
#   correlate keypoints to pixels (Not required)
            #AKA insert column at the start
#   Bring back main

# ---Stdlib---
import sys
from argparse import ArgumentParser
from pathlib import Path
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
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

from utils import ops
from models import Sand

DEFAULT_MODEL_NAME = 'ckpt_G10D' #10 channel
DEFAULT_MODEL_PATH = ROOT/'ckpts'

parser = ArgumentParser('SAND feature extraction demo')
parser.add_argument('--model-name', default=DEFAULT_MODEL_NAME, help='Name of model to load')
parser.add_argument('--model-path', default=DEFAULT_MODEL_PATH, help='Path to directory containing models')
#parser.add_argument('--image-file', default=DEFAULT_IMAGE, help='Path to image to run inference on')

#################### load images
image_path=ROOT/'images'
#image_path='/vol/vssp/datasets/vid+depth/kitti/odometry/dataset/sequences/00'
onlyfiles = [ f for f in listdir(image_path) if isfile(join(image_path,f)) ]
images = np.empty(len(onlyfiles), dtype=object)
print('len(images)',len(images))

#load all images in folder ROOT/'images'
for n in range(0, len(onlyfiles)):
  images[n] = cv.imread( join(image_path,onlyfiles[n]) )
print('len(images)',len(images))


#store key point attributes
key_sand_des = []
class frame_store: #last_frame_keyPoint(x_cord,y_cord,key_sand_des)
    def __init__(self, x_cord, y_cord,keypoints, key_sand_des):

        self.x_cordinate = x_cord
        self.y_cordinate = y_cord
        self.keypoints = keypoints
        self.key_sand_des = key_sand_des

####### FOR ALL IMAGES IN THE FOLDER do SAND and SIFT ###########
for z in range(0, len(images)):

    def get_feat_descrpt():
        args = parser.parse_args()
        device = ops.get_device()
        ckpt_file = Path(args.model_path, args.model_name).with_suffix('.pt')
        #change
        img_file = images[z]

    # Load image & convert to torch format
        img_np = imread(img_file)
        img_torch = ops.img2torch(img_np, batched=True).to(device)

        print(f'Image size (np): {img_np.shape}')
        print(f'Image size (torch): {img_torch.shape}')

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

    #convert to grayscale image
    gray_scale = cv.cvtColor(images[z], cv.COLOR_BGR2GRAY)
    #initialize SIFT object
    sift = cv.SIFT_create()
    keypoints,desc= sift.detectAndCompute(images[z], None)
    x_cord=[]
    y_cord=[]
    for i in range(0,len(keypoints),1):
        x_cord.append(int(keypoints[i].pt[0]))

        y_cord.append(int(keypoints[i].pt[1]))
    #mains()
########################################################
    
    #def mains():
    args = parser.parse_args()
    device = ops.get_device()

    ckpt_file = Path(args.model_path, args.model_name).with_suffix('.pt')
    #img_file = Path(args.image_file)
    img_file = images[z]

    # Load image & convert to torch format
    img_np = images[z]
    img_torch = ops.img2torch(img_np, batched=True).to(device)

    print(f'Image size (np): {img_np.shape}')
    print(f'Image size (torch): {img_torch.shape}')

    # Create & load model (single branch)
    model = Sand.from_ckpt(ckpt_file).to(device)
    model.eval()

    # Run inference
    with torch.no_grad():
        features_torch = model(img_torch)

    # Changes ==> xConvert features into an images we can visualize (by PCA or normalizing)x
    #features_np = ops.fmap2img(features_torch).squeeze(0) torch2 #change fmap2img to torch2np

    features_np = ops.torch2np(features_torch).squeeze(0)

    print(f'SAND Feature size (torch): {features_torch.shape}')
    print(f'SAND Feature size (np): {features_np.shape}')

    ######### pick SIFT features#####
    prev_desc= key_sand_des = []
    key_sand_des = []

    #picking descriptors of SIFT keypoints
    for y,x in zip(y_cord, x_cord): #for the pixels cordinates x and y pick SAND desc
        #key_sand_des[z] = np.append(key_sand_des[z], features_np[y, x])

        key_sand_des.append(features_np[y, x])
    key_sand_des=np.asarray(key_sand_des) #converting to array
    #reshape vector
    key_sand_des=key_sand_des.reshape(-1,10) #length(cordinates) x 10 dimension descriptor
    print('key_sand_des.shape', key_sand_des.shape, '\n ')
    #frame_store(x_cord, y_cord, key_sand_des)
    # matching
    if z==0: #avoid over riding last frame values
        last_frame=frame_store(x_cord, y_cord, keypoints, key_sand_des)
    else: #only match if atleast 2 frames r present
        #BFMatcher with default params
        bf = cv.BFMatcher()
        matches = bf.knnMatch(last_frame.key_sand_des, key_sand_des, k=2)
        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])
        # cv.drawMatchesKnn expects list of lists as matches.
        img3 = cv.drawMatchesKnn(images[z-1], last_frame.keypoints, images[z], keypoints, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img3), plt.show()
    last_frame = frame_store(x_cord, y_cord, keypoints, key_sand_des)
########################

        # if __name__ == '__main__':
        #     main()
