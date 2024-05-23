#Used imshow instead of matplotlib
#Imshow has better image quality than matplotlib
# Import necessary libraries
import numpy as np
import cv2 as cv

# Load image
image_path = "C:\\Users\\JARVIS\\Downloads\\photo.jpg"
#image_path ="C:\\Users\\JARVIS\\Downloads\\DigiKey.jpg"
#image_path ="C:\\Users\\JARVIS\\Downloads\\XRP_Robot-08.jpg"#"C:\Users\JARVIS\Downloads\XRP_Robot-08.jpg"
#image_path = 'C:\\Users\\JARVIS\\Documents\\Tech drive\\Dissertation\\whole code SAND_features\\images\\sample.png'
img = cv.imread(image_path)
#cv.imshow('Corners', img)  # Display the image in a window
# Convert the image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Detect corners in the grayscale image using the Shi-Tomasi method
corners = cv.goodFeaturesToTrack(gray, 150, 0.01, 10)
corners = np.int0(corners)

# Draw circles around detected corners on the original image
for i in corners:
    x, y = i.ravel()  # Flatten the array and get x, y coordinates
    cv.circle(img, (x, y), 6, 250, 2)  # #(image name,<x & y coordinates>, radius, color, thickness of circle line(-1=filled) )

# Save the modified image to disk
#cv.imwrite('C:\\Users\\JARVIS\\Documents\\Tech drive\\Dissertation\\output images\\gftt_xrp08.jpg', img)  # Save the image

# Display the image with corners highlighted using OpenCV
cv.imshow('Corners', img)  # Display the image in a window
cv.waitKey(0)  # Wait for a key press to close the window
cv.destroyAllWindows()  # Close the window
