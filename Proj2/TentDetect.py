"""
ENGO659: Digital Imaging and Applications
Project 2: Counting Tents in a Refugee Camp.

Written by: Veronica Obodozie, Wooju Chung, Alicia Hong
Date: 17-Feb-2025 

This script is the implementation of an algprithm to detect the tents in a refugee camp. It takes the panchromatic image
as an input, and gives a cropped area with the edges of the detected tents highlighted with the number of tents.

File and image type was selected by viewing the histogram.

PRE-PROCESSING
Several Image pre-processing steps were performed to reduce noise and enhance the image as below:
1. The coordinates of region with the least noise and greatest contrast was used to crop region of interest with the PIL package.
this image was saved as a .png file
2. Mathematical Morphology, Dilation, was applied on cropped image using CV2 package.
3. The dilated image was then binarized using thresholding.
4. Following thresholding, a Laplacian of gaussian filter was applied to the binary image to smooth image.

DETECTION
After pre-processing, the tents were detected using the Canny edge detection.

COUNT
The detected tents were counted using the findContour function.


REFERENCES:
reviewing raster files: https://automating-gis-processes.github.io/CSC18/lessons/L6/plotting-raster.html
OpenCVs Python tutorials were very helpful in figuring out the 
    Mathematecal Morphology: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
For counting object method: https://www.geeksforgeeks.org/count-number-of-object-using-python-opencv/

"""

#-----------------------Importing Packages-------------------------#
import cv2
from PIL import Image
import numpy as np
import os
from matplotlib import pyplot as plt

#----------------------Crop Region of Interest-------------------------#
# img_pil = Image.open(r"C:\Users\Veron\OneDrive\GitHub\ENGO659\Proj2\Chad-Mille-QB-2.tif")
img_pil = Image.open(r"/Users/wj/Downloads/ENGO 659/ENGO659/Proj2/Chad-Mille-QB-2.tif")
# img_pil.show()
cropped_image = img_pil.crop((2679, 1038, 3357, 1670))
#(left, top, right, bottom)
# Shows the image in image viewer
cropped_image.save('cropped_image.png')

#----------------------Image Enhancement-------------------------#
# Dilate Image
img = cv2.imread('cropped_image.png')

# Dilate cropped image
# Math Morphology
# Creating appropriate structuring element
strel3x3= cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
# dialated 3x3
dilateMM = cv2.dilate(img, strel3x3)

# Threshold
ret, thresh = cv2.threshold(dilateMM, 127, 255, cv2.THRESH_BINARY)

# Filtering with Laplacian Gaussian
gausBlur = cv2.GaussianBlur(thresh, (5, 5), 0)
# Trying an LoG
LoG = cv2.Laplacian(gausBlur,cv2.CV_8U)

#----------------------Image Detection-------------------------#

# Edge Detection: Canny: https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html? HoughLines?
edges = cv2.Canny(LoG,100,200)
# plt.figure(2)
# plt.subplot(121),plt.imshow(img)
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges)
# plt.title('Canny Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()

#----------------------Counting algoritm-------------------------#
#Count: https://www.geeksforgeeks.org/count-number-of-object-using-python-opencv/
(cnt, hierarchy) = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(3)
cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)
plt.title('Detected: '+ str(len(cnt))+' Tents in Region')
plt.imshow(rgb)
plt.show()
print("Number of tents in region of interest: ", len(cnt))


# # alternative code for getting the number
# for i, c in enumerate(cnt):
#     # 윤곽선
#     cv2.drawContours(rgb, [c], -1, (0, 255, 0), 2)  

#     # 좌표계산해 숫자넣기
#     M = cv2.moments(c)
#     if M["m00"] != 0:  # 면적이 0
#         cx = int(M["m10"] / M["m00"])
#         cy = int(M["m01"] / M["m00"])
#     else:  # 면적이 0인 객체에 대한 임시 좌표
#         cx, cy = c[0][0]  # 객체의 첫째 점
    
#     if i == 48 or i == 51:  #4 9번과 52번 인덱스 48, 51
#         print(f"Object {i+1} - Centroid: ({cx}, {cy})")
#     # cv2.circle(rgb, (cx, cy), 5, (0, 0, 255), -1)  # 원그리기
#     # 번호 표시
#     cv2.putText(rgb, str(i+1), (cx-30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
# plt.figure(5)
# rgb = cv2.resize(rgb, None, fx=1.5, fy=1.5)  # 크기 1.5배 확대
# plt.title('Detected: '+ str(len(cnt))+' Tents in Region')
# plt.imshow(rgb)
# plt.show()
# print("Number of tents in region of interest: ", len(cnt))