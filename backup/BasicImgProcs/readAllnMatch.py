import os
import cv2

def read allimg:
    # specify the path to the folder containing the images
    folder_path = 'C:/path/'

    # create a list to store the images
    images = []

    # iterate over all files in the folder
    for filename in os.listdir(folder_path):
        # check if the file is an image (assuming images have .jpg, .jpeg, .png, or .bmp extensions)
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # read the image using OpenCV
            img = cv2.imread(os.path.join(folder_path, filename))
            # add the image to the list
            images.append(img)

    # now you can use the images list to access all the images in the folder
    for img in images:
        file_name, file_extension = os.path.splitext(file_name)
        siftMatcher(img, file_name,file_extension )


         

#SIFT Matcher
def siftMatcher(img, file_name,file_extension):
    outputImg_path="D:\Aswin\old experiments run\2024_new\matchOut"
    # Create a mirror image by flipping horizontally
    mirror_img = cv.flip(img, 1) #1: Flip horizont 0: Vertical -1: Flip both H and V

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
        if m.distance < 0.55 * n.distance: 
            good_matches.append([m])

    # Select good matched keypoints
    ref_matched_kpts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches])
    sensed_matched_kpts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches])

    # Compute homography
    H, status = cv.findHomography(sensed_matched_kpts, ref_matched_kpts, cv.RANSAC, 5.0)

    # Print inliers and outliers count
    print("No of outliers:", len(status) - np.sum(status))
    print("No of inliers:", np.sum(status))
    inlier_ratio = float(np.sum(status)) / float(len(status))
    print('Inlier ratio:', inlier_ratio)

    # Draw matches
    #img3 = cv.drawMatchesKnn(img1, kp1, mirror_img, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # Draw matches with increased line width
    img3 = cv.drawMatchesKnn(img1, kp1, mirror_img, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), matchesMask=None)
     
    
    
    # get the file name and extension separately
    #file_name, file_extension = os.path.splitext(file_name)

    # specify the new file name
    new_file_name = file_name + '_new' + file_extension

    # specify the full path to the new file
    new_file_path = os.path.join(outputImg_path, new_file_name)

    # save the image to the new file
    cv2.imwrite(new_file_path, img)
    #cv.imwrite('C:\\Users\\JARVIS\\Documents\\Tech drive\\Dissertation\\output images\\DigiKEYmatch.jpg', img3)  # Save the image

