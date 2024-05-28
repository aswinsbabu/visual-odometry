import cv2

# Read the image
img1 = cv2.imread('D:\\Tech drive\\Dissertation\\2024_new\\lena.jpg')

# Check if the image was loaded successfully
if img1 is None:
    print("Error: Could not load image.")
else:
    # Display the image
    cv2.imshow('Matched Features', img1)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()  # Close the window
