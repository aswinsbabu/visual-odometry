#File name: Main2.2.2.1.py
#To do:
#       correlate keypoints to pixels
            #AKA insert column at the start (On progress)
#Done:  Sand key points obtained
#       reshape keyP desc matrix

# ---Stdlib---
import sys
from argparse import ArgumentParser
from pathlib import Path
import cv2 as cv
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
DEFAULT_IMAGE = ROOT/'images'/'sample.png'

parser = ArgumentParser('SAND feature extraction demo')
parser.add_argument('--model-name', default=DEFAULT_MODEL_NAME, help='Name of model to load')
parser.add_argument('--model-path', default=DEFAULT_MODEL_PATH, help='Path to directory containing models')
parser.add_argument('--image-file', default=DEFAULT_IMAGE, help='Path to image to run inference on')

def get_feat_descrpt():
    args = parser.parse_args()
    device = ops.get_device()

    ckpt_file = Path(args.model_path, args.model_name).with_suffix('.pt')
    img_file = Path(args.image_file)

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

    # Changed
    # changed fmap2img to torch2np
    features_np = ops.torch2np(features_torch).squeeze(0)

    # print(f'Feature size (torch): {features_torch.shape}')
    # print(f'Feature size (np): {features_np.shape}')

    # Plot original image & extracted features
    # ax1, ax2 = plt.subplots(2, 1)[1]
    # ax1.set_xticks([]), ax1.set_yticks([])
    # ax2.set_xticks([]), ax2.set_yticks([])
    # ax1.imshow(img_np)
    # ax2.imshow(features_np)
    # plt.show()
    return features_np
    #########################
####################################### SIFT ##########################

#load image
image = cv.imread("sample.png")
#image = ROOT/'images'/'sample.png'
#convert to grayscale image
gray_scale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

#initialize SIFT object
sift = cv.SIFT_create()
keypoints,desc= sift.detectAndCompute(image, None)
x_cord=[]
y_cord=[]
for i in range(0,len(keypoints),1):
 x_cord.append(int(keypoints[i].pt[0]))

 y_cord.append(int(keypoints[i].pt[1]))


########################################################
def main():
    args = parser.parse_args()
    device = ops.get_device()

    ckpt_file = Path(args.model_path, args.model_name).with_suffix('.pt')
    img_file = Path(args.image_file)

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

    # Changes ==> xConvert features into an images we can visualize (by PCA or normalizing)x
    #features_np = ops.fmap2img(features_torch).squeeze(0) torch2 #change fmap2img to torch2np

    features_np = ops.torch2np(features_torch).squeeze(0)

    print(f'SAND Feature size (torch): {features_torch.shape}')
    print(f'SAND Feature size (np): {features_np.shape}')




    ######### pick SIFT features#####

    key_sand_des = []
    #descPixels= [ [],[],[]]#x,y,descriptor

    for y,x in zip(y_cord, x_cord):
        key_sand_des = np.append(key_sand_des, features_np[y, x])

        #print(features_np)
        #key_sand_des=features_np[y,x,:] #2D array


    ############# Print ######################
    print('length of SIFT keypoint x/y_cord=', len(x_cord))
    #print('key_sand_des----', key_sand_des)
    print('len(key_sand_des) is ==>', len(key_sand_des))

    print('key_sand_des', key_sand_des)
    print('B4 reshape key_sand_des.ndim=', key_sand_des.ndim)

    #reshape vector
    key_sand_des=key_sand_des.reshape(len(x_cord),10) #length(cordinates) x 10 dimension

    #print('Reshaped key_sand_des.ndim', key_sand_des.ndim,'\n \t\t shape =',key_sand_des.shape)
    #print('reshaped key_sand_des.shape', key_sand_des.shape)
    #print('Reshaped key_sand_des', key_sand_des)
    ########################

    #matching
    #key points and descriptors
if __name__ == '__main__':
    main()
