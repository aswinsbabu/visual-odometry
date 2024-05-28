import os
import cv2 as cv
import numpy as np

# Load images
#Clr_img1 = cv.imread('D:\\Tech drive\\Dissertation\\2024_new\\InputImg\\cross_ref.jpg')
#img1 = cv.imread("D:\\Tech drive\\Dissertation\\2024_new\\InputImg\\cross_ref.jpg", cv2.IMREAD_GRAYSCALE)

#save image path
save_path ="D:\\Tech drive\\Dissertation\\2024_new\\output images\\matchOut\\siftBF" #D:\Tech drive\Dissertation\2024_new\output images\matchOut\siftBF

#save_path = "C:\\Users\\JARVIS\\Documents\\Tech drive\\Dissertation\\2024_new\\output images\\matchOut"
folder_path = "D:\\Tech drive\\Dissertation\\2024_new\\InputImg"

def read_allimg():
    # specify the path to the folder containing the images
    
    # create a list to store the images and their filenames
    images = []

    # iterate over all files in the folder
    for filename in os.listdir(folder_path):
        # check if the file is an image (assuming images have .jpg, .jpeg, .png, or .bmp extensions)
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # read the image using OpenCV
            img = cv.imread(os.path.join(folder_path, filename))
            # add the image and its filename to the list
            images.append((img, filename))

    # now you can use the images list to access all the images in the folder
    for img, filename in images:
        file_name, file_extension = os.path.splitext(filename)
        Match_feature(img,file_name, file_extension)

#def GFTT()

# SIFT Matcher
def Match_feature(img, file_name, file_extension): #file_name & f_--extension bypassed to next function
    

    # Create a mirror image by flipping horizontally
    mirror_img = cv.flip(img, 1)  # 1: Flip horizontally, 0: Vertical, -1: Flip both H and V

########################################################################################
    # Initiate SIFT detector
    sift = cv.SIFT_create()

    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img, None)
    kp2, des2 = sift.detectAndCompute(mirror_img, None)

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.55 * n.distance:
            good_matches.append([m])

    # Select good matched keypoints
    #ref_matched_kpts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches])
    #sensed_matched_kpts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches])

##############################################################################################

    # Draw matches with increased line width
    img3 = cv.drawMatchesKnn(img, kp1, mirror_img, kp2, good_matches, None, 
                             flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


    display_imgs(img3)
    save_imgs(img3,save_path,file_name, file_extension)

def save_imgs(img3, save_path,file_name, file_extension):
    # Save the modified image to disk
    new_file_name = file_name + '_newx' + file_extension
    new_file_path = os.path.join(save_path, new_file_name) #
    cv.imwrite(new_file_path, img3)  # Save the image
    
def display_imgs(img3):
    # Display the image with OpenCV
    cv.imshow('Matched Features', img3)
    cv.waitKey(2000)
    cv.destroyAllWindows()

read_allimg()