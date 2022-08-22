#File name: Main2.3.2 py ==> Parent 2.3
#change: [z] added
#         removed main
#To do:
#   matching
#
#           loop n times (On progress)
#   correlate keypoints to pixels (Not required)
            #AKA insert column at the start
#   Bring back main
#Done:  Sand key points obtained
#       reshape keyP desc matrix
#   matching
#       Insert 2 images and compute
#           display all n images (done)

# ---Stdlib---
import sys
from argparse import ArgumentParser
from pathlib import Path
import cv2 as cv
import glob  #image read
import numpy as np

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
from os import listdir
from os.path import isfile, join

mypath=ROOT/'images'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = np.empty(len(onlyfiles), dtype=object)

#load all images in folder ROOT/'images'
for n in range(0, len(onlyfiles)):
  images[n] = cv.imread( join(mypath,onlyfiles[n]) )
print('len(images)',len(images))

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

    key_sand_des = []
    #descPixels= [ [],[],[]]#x,y,descriptor
    #for image in images:
    for y,x in zip(y_cord, x_cord):
        #current image= image[z]
        #key_sand_des[z] = np.append(key_sand_des[z], features_np[y, x])

        key_sand_des.append(features_np[y, x])
    key_sand_des=np.asarray(key_sand_des) #converting to array
    #print(features_np)
    #key_sand_des[z]=features_np[y,x,:] #2D array


    ############# Print ######################
    print('length of SIFT keypoint x/y_cord=', len(x_cord))
    #print('key_sand_des[z]----', key_sand_des)
    print('len(key_sand_des[z]) is ==>', len(key_sand_des))

    print('key_sand_des[z]', key_sand_des)
    print('B4 reshape key_sand_des[z].ndim=', key_sand_des.ndim)

    #reshape vector
    key_sand_des=key_sand_des.reshape(-1,10) #length(cordinates) x 10 dimension
    print('key_sand_des.shape', key_sand_des.shape)
    #print('Reshaped key_sand_des.ndim', key_sand_des.ndim,'\n \t\t shape =',key_sand_des.shape)
    #print('reshaped key_sand_des.shape', key_sand_des.shape)
    #print('key_sand_des', key_sand_des)

        ########################

        #matching
        #key points(SIFT) and descriptors(SAND)
        # if __name__ == '__main__':
        #     main()





