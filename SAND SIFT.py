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


DEFAULT_MODEL_NAME = 'ckpt_G10D'
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

    # Convert features into an images we can visualize (by PCA or normalizing)
    features_np = ops.fmap2img(features_torch).squeeze(0)

    print(f'Feature size (torch): {features_torch.shape}')
    print(f'Feature size (np): {features_np.shape}')

    # Plot original image & extracted features
    ax1, ax2 = plt.subplots(2, 1)[1]
    ax1.set_xticks([]), ax1.set_yticks([])
    ax2.set_xticks([]), ax2.set_yticks([])
    ax1.imshow(img_np)
    ax2.imshow(features_np)
    plt.show()
    return features_np
    #########################
####################################### SIFT ##########################

#load image
#image = cv.imread("lena.jpg")
image = ROOT/'images'/'sample.png'
#convert to grayscale image
gray_scale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

#initialize SIFT object
sift = cv.SIFT_create()
#sift = cv.xfeatures2d.SIFT_create()

#detect keypoints

keypoints,desc= sift.detectAndCompute(image, None)
x_cord=[]
y_cord=[]
for i in range(0,len(keypoints),1):
 x_cord.append(int(keypoints[i].pt[0]))

 y_cord.append(int(keypoints[i].pt[1]))

print ('length of cord=',len(x_cord))
#print ('length of y_cord=',len(y_cord))
print(x_cord)
print(y_cord)
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

    # Convert features into an images we can visualize (by PCA or normalizing)
    features_np = ops.fmap2img(features_torch).squeeze(0)

    print(f'Feature size (torch): {features_torch.shape}')
    print(f'Feature size (np): {features_np.shape}')

    # Plot original image & extracted features
    # ax1, ax2 = plt.subplots(2, 1)[1]
    # ax1.set_xticks([]), ax1.set_yticks([])
    # ax2.set_xticks([]), ax2.set_yticks([])
    # ax1.imshow(img_np)
    # ax2.imshow(features_np)
    #plt.show()
    print('376*1241*3=',376*1241*3)
    print('feature.np.size/no=',features_np.size) #nd array
    print('type(feature_np)=',type(features_np))
    print('feature.shape=', features_np.shape)
    ######### pick SIFT features#####
    # fruits = ["apple", "banana", "cherry"]
    # for x in fruits:
    #     print(x)
    row = x_cord
    col =y_cord

    for x in row and y in col:
        print(features_np[x,y])
        key_sand_des=features_np[x,y] #2D array
    ########################

if __name__ == '__main__':
    main()
