import cv2
import numpy as np
# read image
img = cv2.imread('/vol/research/visual_localization/experiments/tony.jpg')
#img = cv2.imread('lena.jpg')
# show image
cv2.imshow('lena', img)

# waits for user to press any key
# (this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0)#wait infinitely ,5 means 5 seconds

# closing all open windows
cv2.destroyAllWindows()

# cv2.waitKey(0)  # waits until a key is pressed
# cv2.destroyAllWindows()  # destroys the window showing image
