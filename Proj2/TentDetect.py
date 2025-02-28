"""
ENGO659: Digital Imaging and Applications
Project 2: Counting Tents in a Refugee Camp.

Written by: Veronica Obodozie, Wooju Chung, Alicia Hong
Date: 28-Feb-2025 

This script is the implementation of an algprithm to detect the tents in a refugee camp. It takes the panchromatic image
as an input, and gives a cropped area with the edges of the detected tents highlighted with the number of tents.

File and image type was selected by viewing the histogram.

PRE-PROCESSING
Several Image pre-processing steps were performed to reduce noise and enhance the image as below:
1. The coordinates of region with the least noise and greatest contrast was used to crop region of interest with the PIL package.
this image was saved as a .png file
2. Mathematical Morphology, Dilation, was applied on cropped image using CV2 package.
3. The dilated image was then binarized using thresholding.
4. Following thresholding, a Gaussian filter was applied to the binary image to smooth image.

DETECTION
After pre-processing, the tents were detected using the Canny edge detection.

COUNT
The detected tents were counted using the findContour function.

INPUT
OUTPUT: 1.Detected tents highlighted
        2 Total number of Tents counted

REFERENCES:
Used for initial reviewing raster files: https://automating-gis-processes.github.io/CSC18/lessons/L6/plotting-raster.html
OpenCVs Python tutorials were very helpful in figuring out the 
    Mathematecal Morphology: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    Canny Edge Detection: https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html? 
For counting object method: https://www.geeksforgeeks.org/count-number-of-object-using-python-opencv/

"""

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
# The contours are displayed on the image, along with the number of tents in the title.
plt.figure(1)
cv2.drawContours(img, cnt, -1, (0, 255, 0), 2)
plt.title('Detected: '+ str(len(cnt))+' Tents in Region')
plt.imshow(img)
plt.show()
# Printing out the number of tents
print("Number of tents in region of interest: ", len(cnt))
