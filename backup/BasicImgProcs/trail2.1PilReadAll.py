import os
import cv2 as cv
import numpy as np
from PIL import Image

# Define paths
output_folder_path = "D:\\Tech drive\\Dissertation\\2024_new\\output images\\matchOut\\dip"
input_folder_path = "D:\\Tech drive\\Dissertation\\2024_new\\InputImg"
tag = 'XRP_one_0.65'
new_dpi = 300  # Desired DPI
images = []  # Image storage

# Read all images
def read_all_images():
    """
    Read all images from the specified folder, resize them, and match features.
    """
    # Iterate over all files in the folder
    for filename in os.listdir(input_folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            img = cv.imread(os.path.join(input_folder_path, filename))
            if img is not None:
                images.append((img, filename))

# Display images
def display_image(img_path):
    """
    Display the image using OpenCV.
    """
    img = cv.imread(img_path)
    if img is not None:
        cv.imshow('Image', img)
        cv.waitKey(1000)
        cv.destroyAllWindows()
    else:
        print(f"Failed to load image from path: {img_path}")

# Change the DPI using Pillow
def change_dpi():
    i = 1
    for img, filename in images:
        try:
            pil_img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
            new_filename = f"{os.path.splitext(filename)[0]}_{tag}_{i}.jpg"
            output_image_path = os.path.join(output_folder_path, new_filename)
            pil_img.save(output_image_path, dpi=(new_dpi, new_dpi))
            i += 1
        except Exception as e:
            print(f"Failed to change DPI for {filename}: {e}")

##################### Main ###########################
read_all_images()
change_dpi()

# Display a sample image from the output to verify
sample_output_image_path = os.path.join(output_folder_path, os.listdir(output_folder_path)[0])
display_image(sample_output_image_path)