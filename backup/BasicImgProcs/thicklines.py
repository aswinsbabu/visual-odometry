import cv2
import numpy as np
def draw_matches(img1, kp1, img2, kp2, matches, line_width):
    """
    Draw matches between two images.

    Args:
        img1: The first image.
        kp1: The keypoints in the first image.
        img2: The second image.
        kp2: The keypoints in the second image.
        matches: The matches between the keypoints.
        line_width: The width of the lines connecting the matches.

    Returns:
        The image with the drawn matches.
    """
    # Create a copy of the images to draw on
    img_matches = np.concatenate((img1, img2), axis=1)

    # Draw the matches
    for match in matches:
        x1, y1 = kp1[match.queryIdx].pt
        x2, y2 = kp2[match.trainIdx].pt + (img1.shape[1], 0)
        cv2.line(img_matches, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), line_width)

    return img_matches

# Load images
img1 = cv2.imread('D:\\Tech drive\\Dissertation\\2024_new\\lena.jpg', cv2.IMREAD_GRAYSCALE)

 # Create a mirror image by flipping horizontally
img2 = cv2.flip(img1, 1)  # 1: Flip horizontally, 0: Vertical, -1: Flip both H and V

# Detect keypoints and compute descriptors
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Match the keypoints
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test to select good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.5 * n.distance:
        good_matches.append(m)

# Draw the matches with a custom line width
img_matches = draw_matches(img1, kp1, img2, kp2, good_matches, line_width=5)

# Display the matches
cv2.imshow('SIFT Feature Matching', img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()