
#-----------------------Importing Packages-------------------------#
# A requirements file has been attached as well. These packages are needed for the code to function.
import cv2
from PIL import Image
import numpy as np
import os
from matplotlib import pyplot as plt

#----------------------Crop Region of Interest-------------------------#
# This code section is for reference only, it was used to crop the larger satellite image to the regoin of interest.
# It uses the pillow package to open and crop the image based on the filepath provided within the code.
# img_pil = Image.open(r"C:\Users\Veron\OneDrive\GitHub\ENGO659\Proj2\Chad-Mille-QB-2.tif")
# img_pil = Image.open(r"/Users/wj/Downloads/ENGO 659/ENGO659/Proj2/Chad-Mille-QB-2.tif")
# img_pil.show()
# cropped_image = img_pil.crop((1580, 1731, 2040, 2187))
#(left, top, right, bottom)
# Shows the image in image viewer
# cropped_image.save('cropped_image.png')

#-----------------------------Image Enhancement-----------------------------#
# Certain steps were taked to enhance the image prior to detection
# Read image using Open Source Computer Vision
img = cv2.imread('cropped_image.png')
# img = cv2.resize(img, None, fx=1.5, fy=1.5
#                  , interpolation=cv2.INTER_CUBIC
#                  )  # 크기 1.5배 확대

# 1. Dilate cropped image
# Math Morphology, dilation, method used to increase tent sizes

# Creating 3 x 3 structuring element
strel3x3= cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
# Image then dialated using 3x3 strel
dilateMM = cv2.dilate(img, strel3x3)

# 2. Threshold
# Applying thresholding to dilated image. This binarizes the image.
# The threshold has a lower bound of 94 and upper bound of 185, based on dark and light tent histogram values.
ret, thresh = cv2.threshold(dilateMM, 94, 185, cv2.THRESH_BINARY)

# Filtering with Gaussian
# A 5x5 gaussian filter was used to reduce noise in the binary image. Necessary when using edge detection version.
gausBlur = cv2.GaussianBlur(thresh, (5, 5), 0)

#----------------------------------Image Detection----------------------------------------#
# Canny Edge Detection was used to determine tents.
edges = cv2.Canny(gausBlur,100,200)

#-------------------------------------Counting algoritm------------------------------------#
# Counting detected tents was done using the built-in findContours method
(cnt, hierarchy) = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# The contours are displayed on the image, along with the number of tents in the tittle.
# plt.figure(1)
# cv2.drawContours(img, cnt, -1, (0, 255, 0), 2)
# plt.title('Detected: '+ str(len(cnt))+' Tents in Region')
# plt.imshow(img)
# plt.show()
# # Printing out the number of tents
# print("Number of tents in region of interest: ", len(cnt))


# alternative code for getting the number
for i, c in enumerate(cnt):
    # 윤곽선
    cv2.drawContours(img, [c], -1, (0, 255, 0), 2)  

    # 좌표계산해 숫자넣기
    M = cv2.moments(c)
    if M["m00"] != 0:  # 면적이 0
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:  # 면적이 0인 객체에 대한 임시 좌표
        cx, cy = c[0][0]  # 객체의 첫째 점
        print(i+1,cx,cy,"면적0인거")
    
    if i == 98-1 or i ==84-1 or i ==89-1 or i ==93-1:  # 84,89,98
        print(f"Object {i+1} - Centroid: ({cx}, {cy})")
    cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)  # 원그리기
    # 번호 표시
    cv2.putText(img, str(i+1), (cx-10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1) # cx는 좌표위치, 0.2 가 폰트크기, 1이 굵기
plt.figure(5)
# img = cv2.resize(img, None, fx=10, fy=10)  # 크기 1.5배 확대
plt.title('Detected: '+ str(len(cnt))+' Tents in Region')
plt.imshow(img)
plt.show()
print("Number of tents in region of interest: ", len(cnt))