The provided code that reads images from a folder, resizes them, matches features using SIFT (Scale-Invariant Feature Transform), 
and draws matches with different colors for both the lines and the keypoints.

Code Breakdown
python
Copy code
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
Imports: Necessary libraries are imported.
Paths: Define the paths for input images and the output folder.
Tag and DPI: Define a tag to append to output filenames and the desired DPI for images.
Ensure Output Directory Exists: Create the output directory if it doesn't already exist.
python
Copy code
def resize_image(image):
    """
    Resize the image to half of its original dimensions.
    """
    original_height, original_width = image.shape[:2]
    new_width = int(original_width // 2)
    new_height = int(original_height // 2)
    resized_image = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_LINEAR)
    return resized_image
resize_image Function: This function resizes the input image to half of its original dimensions using cv.resize and returns the resized image.
python
Copy code
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
read_all_images Function: This function reads all images from a specified folder.
Iterates over all files in the folder.
Checks if the file is an image based on its extension.
Reads the image using cv.imread and appends it to the images list.
For each image, it resizes the image and calls the match_features function.
python
Copy code
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

    # Manually draw matches with increased line thickness and different colors
    img3 = draw_matches(img, kp1, mirror_img, kp2, good_matches, line_thickness=2, circle_radius=5)

    # Save and display the results
    save_images(img3, save_path, file_name, file_extension)
    display_images(img3)
match_features Function: This function matches features between an image and its horizontally flipped version.
Flips the image using cv.flip.
Uses SIFT to detect and compute keypoints and descriptors.
Uses BFMatcher to find the best matches between descriptors.
Applies a ratio test to filter out good matches.
Calls the draw_matches function to draw matches with increased line thickness and different colors.
Saves and displays the result.
python
Copy code
def draw_matches(img1, kp1, img2, kp2, matches, line_thickness, circle_radius):
    """
    Manually draw matches with increased line thickness and different colors.
    """
    # Create a combined image to draw on
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    combined_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    combined_img[:h1, :w1] = img1
    combined_img[:h2, w1:w1 + w2] = img2

    # Generate random colors
    np.random.seed(42)  # For reproducibility
    colors = np.random.randint(0, 255, (len(matches), 3)).tolist()

    # Draw the matches
    for idx, match in enumerate(matches):
        pt1 = (int(kp1[match[0].queryIdx].pt[0]), int(kp1[match[0].queryIdx].pt[1]))
        pt2 = (int(kp2[match[0].trainIdx].pt[0] + w1), int(kp2[match[0].trainIdx].pt[1]))
        color = colors[idx]
        cv.line(combined_img, pt1, pt2, color, thickness=line_thickness)
        cv.circle(combined_img, pt1, circle_radius, color, thickness=-1)  # Filled circle
        cv.circle(combined_img, pt2, circle_radius, color, thickness=-1)  # Filled circle

    return combined_img
draw_matches Function: This function manually draws matches with increased line thickness and different colors.
Combines the two images (original and mirrored) side by side.
Generates random colors for each match.
Draws lines and circles for each match using the generated colors.
python
Copy code
def save_images(img, save_path, file_name, file_extension):
    """
    Save the processed image to the specified path.
    """
    new_file_name = file_name + tag + file_extension
    new_file_path = os.path.join(save_path, new_file_name)
    cv.imwrite(new_file_path, img)
save_images Function: This function saves the processed image to the specified path.
python
Copy code
def display_images(img):
    """
    Display the image using OpenCV.
    """
    cv.imshow('Matched Features', img)
    cv.waitKey(1000)
    cv.destroyAllWindows()
display_images Function: This function displays the image using OpenCV.
python
Copy code
# Run the function to read and process all images
read_all_images()
Main Execution: This calls the read_all_images function to start the process.

Summary
The code reads images from a specified folder, resizes them to half of their original dimensions, uses SIFT to detect keypoints and descriptors,
matches features between the original and horizontally flipped images, and draws lines and circles with unique colors for each match. 
The results are then saved and displayed.

