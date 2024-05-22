import cv2 as cv

# Load image
image_path = "C:\\Users\\JARVIS\\Downloads\\photo.jpg"
img = cv.imread(image_path)

# Create a mirror image by flipping horizontally
mirror_img = cv.flip(img, 1)

# Display the original and mirror images
cv.imshow('Original Image', img)
cv.imshow('Mirror Image', mirror_img)
cv.waitKey(0)
cv.destroyAllWindows()
