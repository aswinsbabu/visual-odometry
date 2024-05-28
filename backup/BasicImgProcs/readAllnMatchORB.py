import os
import cv2 as cv
import numpy as np

matchImg_path = "C:\\Users\\JARVIS\\Documents\\Tech drive\\Dissertation\\2024_new\\output images\\matchOut"
#folder_path = "C:\\Users\\JARVIS\\Documents\\Tech drive\\Dissertation\\2024_new\\InputImg\\subset"
folder_path = "C:\\Users\\JARVIS\\Documents\\Tech drive\\Dissertation\\2024_new\\InputImg"

def read_allimg():
    # Create a list to store the images and their filenames
    images = []

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is an image (assuming images have .jpg, .jpeg, .png, or .bmp extensions)
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # Read the image using OpenCV
            img = cv.imread(os.path.join(folder_path, filename))
            # Add the image and its filename to the list
            images.append((img, filename))

    # Now you can use the images list to access all the images in the folder
    for img, filename in images:
        file_name, file_extension = os.path.splitext(filename)
        Match_feature(img, file_name, file_extension)

def Match_feature(img, file_name, file_extension): 
    # Create a mirror image by flipping horizontally
    mirror_img = cv.flip(img, 1)  # 1: Flip horizontally, 0: Vertical, -1: Flip both H and V

    # ORB initialization
    orb = cv.ORB_create()

    # Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img, None)
    kp2, des2 = orb.detectAndCompute(mirror_img, None)

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.55 * n.distance:
            good_matches.append([m])

    # Draw matches with increased line width
    img3 = cv.drawMatchesKnn(img, kp1, mirror_img, kp2, good_matches, None, 
                             flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Draw thicker lines for the matches
    for match in good_matches:
        pt1 = tuple(map(int, kp1[match[0].queryIdx].pt))
        pt2 = (int(kp2[match[0].trainIdx].pt[0] + img.shape[1]), int(kp2[match[0].trainIdx].pt[1]))  # Offset x by width of img
        cv.line(img3, pt1, pt2, (0, 255, 0), 5)  # Change 10 to any thickness you want

    display_imgs(img3)
    #save_imgs(img3, matchImg_path, file_name, file_extension)

def save_imgs(img3, save_path, file_name, file_extension):
    # Save the modified image to disk
    new_file_name = file_name + '_new' + file_extension
    new_file_path = os.path.join(save_path, new_file_name)
    cv.imwrite(new_file_path, img3)  # Save the image
    
def display_imgs(img3):
    # Display the image with OpenCV
    cv.imshow('Matched Features', img3)
    cv.waitKey(1500)  # Display each image for 1.5 seconds
    cv.destroyAllWindows()

read_allimg()
