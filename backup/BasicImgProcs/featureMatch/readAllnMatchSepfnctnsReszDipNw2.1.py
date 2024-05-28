import os
import cv2 as cv
#import numpy as np
#from PIL import Image

# Define paths
save_path = "D:\\Tech drive\\Dissertation\\2024_new\\output images\\matchOut\\siftBF\\sizeBy2\\by2qlty0.6"
folder_path = "D:\\Tech drive\\Dissertation\\2024_new\\InputImg\\subset" #D:\\Tech drive\\Dissertation\\2024_new\\output images\\matchOut\\siftBF\\sizeBy2\\by2qlty0.6
tag = 'XRP_one_0.5__2'
#new_dpi = 300  # Desired DPI

# Change the DPI using Pillow
# def change_dpi(input_image_path, output_image_path, new_dpi):
#     with Image.open(input_image_path) as img:
#         img.save(output_image_path, dpi=(new_dpi, new_dpi))

def resize_image(image):
    """
    Resize the image to half of its original dimensions.
    """
    original_height, original_width = image.shape[:2]
    new_width = int(original_width // 2)
    new_height = int(original_height // 2)
    resized_image = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_LINEAR)
    display_images(resized_image)
    #display_images(image)
    return resized_image

def read_all_images():
    """
    Read all images from the specified folder, resize them, and match features.
    """
    images = []

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            img = cv.imread(os.path.join(folder_path, filename))
            if img is not None:
                images.append((img, filename))

    # Process each image
    for img, filename in images:
        file_name, file_extension = os.path.splitext(filename)
        resized_image = resize_image(img)
        match_features(resized_image, file_name, file_extension)

def match_features(img, file_name, file_extension):
    """
    Match features between the image and its horizontally flipped version.
    """
    mirror_img = cv.flip(img, 1)

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
        if m.distance < 0.5 * n.distance:
            good_matches.append([m])

    # Draw matches with increased line width
    img3 = cv.drawMatchesKnn(img, kp1, mirror_img, kp2, good_matches, None, 
                             flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Save and display the results
    save_images(img3, save_path, file_name, file_extension)

def save_images(img, save_path, file_name, file_extension):
    """
    Save the processed image to the specified path.
    """
    new_file_name = file_name + tag + file_extension
    new_file_path = os.path.join(save_path, new_file_name)
    cv.imwrite(new_file_path, img)

def display_images(img):
    """
    Display the image using OpenCV.
    """
    cv.imshow('Matched Features', img)
    cv.waitKey(1000)
    cv.destroyAllWindows()
    return 0

# Run the function to read and process all images
read_all_images()
