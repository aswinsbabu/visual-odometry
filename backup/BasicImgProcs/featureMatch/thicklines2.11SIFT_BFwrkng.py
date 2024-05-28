import os
import cv2 as cv
import numpy as np

# Define paths
save_path = "D:\\Tech drive\\Dissertation\\2024_new\\output images\\matchOut\\siftBF\\sizeBy2\\by2qlty0.6"
folder_path = "D:\\Tech drive\\Dissertation\\2024_new\\InputImg\\subset"
tag = 'XRP_one_0.6__2'
new_dpi = 300  # Desired DPI

# Ensure output directory exists
os.makedirs(save_path, exist_ok=True)

def resize_image(image):
    """
    Resize the image to half of its original dimensions.
    """
    original_height, original_width = image.shape[:2]
    new_width = int(original_width // 2)
    new_height = int(original_height // 2)
    resized_image = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_LINEAR)
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
        if m.distance < 0.65 * n.distance:
            good_matches.append([m])

    # Manually draw matches with increased line thickness
    img3 = draw_matches(img, kp1, mirror_img, kp2, good_matches, line_thickness=2)

    # Save and display the results
    save_images(img3, save_path, file_name, file_extension)
    display_images(img3)

def draw_matches(img1, kp1, img2, kp2, matches, line_thickness):
    """
    Manually draw matches with increased line thickness.
    """
    # Create a combined image to draw on
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    combined_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    combined_img[:h1, :w1] = img1
    combined_img[:h2, w1:w1 + w2] = img2

    # Draw the matches
    for match in matches:
        pt1 = (int(kp1[match[0].queryIdx].pt[0]), int(kp1[match[0].queryIdx].pt[1]))
        pt2 = (int(kp2[match[0].trainIdx].pt[0] + w1), int(kp2[match[0].trainIdx].pt[1]))
        cv.line(combined_img, pt1, pt2, (0, 255, 0), thickness=line_thickness)

    return combined_img

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

# Run the function to read and process all images
read_all_images()
