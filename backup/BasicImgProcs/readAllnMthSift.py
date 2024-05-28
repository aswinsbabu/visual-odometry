import os
import cv2 as cv
import numpy as np

def read_allimg():
    # specify the path to the folder containing the images
    folder_path = "C:\\Users\\JARVIS\\Documents\\Tech drive\\Dissertation\\2024_new\\InputImg"

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
        siftMatcher(img, file_name, file_extension)

# SIFT Matcher
def siftMatcher(img, file_name, file_extension):
    outputImg_path = "C:\\Users\\JARVIS\\Documents\\Tech drive\\Dissertation\\2024_new\\output images\\matchOut"

    # Create a mirror image by flipping horizontally
    mirror_img = cv.flip(img, 1)  # 1: Flip horizontally, 0: Vertical, -1: Flip both H and V

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
    ref_matched_kpts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches])
    sensed_matched_kpts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches])


    # Draw matches with increased line width
    img3 = cv.drawMatchesKnn(img, kp1, mirror_img, kp2, good_matches, None, 
                             flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Save the modified image to disk
    new_file_name = file_name + '_new' + file_extension
    new_file_path = os.path.join(outputImg_path, new_file_name)
    cv.imwrite(new_file_path, img3)  # Save the image

    # Display the image with OpenCV
    cv.imshow('Matched Features', img3)
    cv.waitKey(0)
    cv.destroyAllWindows()

read_allimg()
