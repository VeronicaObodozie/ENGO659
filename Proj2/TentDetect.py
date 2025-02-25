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
# img_pil = Image.open(r"/Users/wj/Downloads/ENGO 659/ENGO659/Proj2/Chad-Mille-QB-2.tif")
# img_pil.show()
# cropped_image = img_pil.crop((1580, 1731, 2040, 2187))
#(left, top, right, bottom)
# Shows the image in image viewer
# cropped_image.save('cropped_image.png')

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
ret, thresh = cv2.threshold(dilateMM, 94, 185, cv2.THRESH_BINARY)

# Filtering with Laplacian Gaussian
gausBlur = cv2.GaussianBlur(thresh, (5, 5), 0)

#----------------------Image Detection-------------------------#
# Edge Detection: Canny: https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html? HoughLines?
edges = cv2.Canny(gausBlur,100,200)

#----------------------Counting algoritm-------------------------#
#Count: https://www.geeksforgeeks.org/count-number-of-object-using-python-opencv/
(cnt, hierarchy) = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
plt.figure(1)
cv2.drawContours(img, cnt, -1, (0, 255, 0), 2)
plt.title('Detected: '+ str(len(cnt))+' Tents in Region')
plt.imshow(img)
plt.show()
print("Number of tents in region of interest: ", len(cnt))