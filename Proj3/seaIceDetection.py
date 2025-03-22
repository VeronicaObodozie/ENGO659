"""
ENGO659: Digial Imaging and Applications
Project 3: Sea Ice Motion Estimation 

Written by: Alicia Hong, Veronica Obodozie, Wooju Chung 
Date: March 18, 2025

This script is the implementation of an algorithm to estimate the sea ice motion. Two consecutive images are chosen from the dataset containing eight MODIS images 
and the motion of the sea ice pack between the two images is tracked by detecting features in the first image and applying feature-guided area matching in the second image. 

PRE-PROCESSING 
    1) Convert binary imags in .im format to JPEG  and save the images
    2) Convert the images to grayscale images (used for corner detection)
    3) Apply Guassian Blurring to the images (improves performance of template matching)

CORNER (FEATURE) DETECTION
    1)  There is a region in the upper left section that displays higher contrast due to the open water. To prevent the algorithm from returning corners concentrated 
        in this region, the image is divided into four tiles. For each of the tiles, a mask for the tile is created to be used as an input parameter in the 
        goodFeaturesToTrack function to specify the region of interest (ROI). 
    2)  For each of the mask, goodFeaturesToTrack function is used to determine strong corners. 
    3)  Corners detected in each of the four masks are combined into one list. 

FEATURE-GUIDED AREA MATCHING
For each of the corners detected in Image 1, the following steps are performed:
    1)  A 3x3 template using the corner location as the centre is created using Image 1. 
    2)  A 12x12 search window is defined in Iamge 2 with the corner location in the centre. 
    3)  Template matching is performed between the template and the search window using normalized correlation coefficient method. 
    4)  The x,y coordinate of the matched point is appended to the matches list if the result value is above the threshold value of 0.9, 
        where 1 indicates a perfect match.

OUTPUT GENERATION
    1)  On Image 1, the motion of the sea ice is visualized by plotting the detected corners from Image 1 (green circles), the matched corners (magenta circles)
        and the vector between the two corners (blue line). The image is saved as a JPEG file. 
    2)  The x,y coordinates of the detected corners in Image 1 and the matched corners in Image 2 are saved to a csv file. 

REFERENCES:
    OpenCV: Template Matching - https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html
    OpenCV: Feature Detection - https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html
"""

import cv2
import numpy as np
import csv

# region Pre-Procecssing ----------------------------------------------------------------------------------------------------------

# Convert binary image and save as jpg
with open('2007135.im', 'rb') as f:
    image_data1 = np.fromfile(f, dtype=np.uint8)

image1_b = image_data1.reshape((960, 720, 1), order='F')
cv2.imwrite('2007135_out.jpg', image1_b)

with open('2007136.im', 'rb') as f:
    image_data2 = np.fromfile(f, dtype=np.uint8)

image2_b = image_data2.reshape((960, 720, 1), order='F')
cv2.imwrite('2007136_out.jpg', image2_b)

image1 = cv2.imread('2007135_out.jpg')
image2 = cv2.imread('2007136_out.jpg')

# Convert image to grayscale
image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Gaussian Blurring 
blurred_img1 = cv2.GaussianBlur(image1_gray, (5, 5), 0)
blurred_img2 = cv2.GaussianBlur(image2_gray, (5, 5), 0)

# endregion ------------------------------------------------------------------------------------------------------------------------

# region Corner Detection -----------------------------------------------------------------------------------------------------------

# Create a blank image (for mask creation)
blank_image = np.zeros((960, 720), dtype=np.uint8)
# Define image dimension and midpoints 
height, width = 960, 720
mid_height, mid_width = height//2, width//2

# Create mask that defines the region of interest (ROI) where corners can be detected. 
# Tile 1 (Top-Left)
mask1 = np.zeros_like(blank_image) # all pixels are 0 (black)
mask1[0:mid_height, 0:mid_width] = 255 # Set the rectangular region pixels to 255 (white)
# Tile 2 (Top-Right)
mask2 = np.zeros_like(blank_image)
mask2[0:mid_height, mid_width:width] = 255
# Tile 3 (Bottom-Left)
mask3 = np.zeros_like(blank_image)
mask3[mid_height:height, 0:mid_width] = 255
# Tile 4 (Bottom-Right)
mask4 = np.zeros_like(blank_image)
mask4[mid_height:height, mid_width:width] = 255

# Perform Corner Detection using goodFeaturesToTrack function 
corners1 = cv2.goodFeaturesToTrack(image1_gray, maxCorners=70, qualityLevel=0.1, minDistance=7, useHarrisDetector=False, mask=mask1) 
corners2 = cv2.goodFeaturesToTrack(image1_gray, maxCorners=350, qualityLevel=0.1, minDistance=7, useHarrisDetector=True, mask=mask2) 
corners3 = cv2.goodFeaturesToTrack(image1_gray, maxCorners=70, qualityLevel=0.1, minDistance=7, useHarrisDetector=False, mask=mask3) 
corners4 = cv2.goodFeaturesToTrack(image1_gray, maxCorners=20, qualityLevel=0.1, minDistance=7, useHarrisDetector=False, mask=mask4) 

# Combine the corners from the four ROIs
corners = np.vstack((corners1, corners2, corners3, corners4))

# endregion --------------------------------------------------------------------------------------------------------------------------

# region Feature-Guided Area Matching -------------------------------------------------------------------------------------------------
patch_size1 = 6 # patch size for the template 
patch_size2 = 24 # patch size for defining the search neighbourhood 
matches = []

for corner in corners:
    x, y = corner.ravel()
    x, y = int(x), int(y)

    # Create the template using feature location from corners detected in Image 1. 
    template_ymin, template_ymax = y-patch_size1//2, y+patch_size1//2
    template_xmin, template_xmax = x-patch_size1//2, x+patch_size1//2
    # Ensure template is within valid range
    template_xmin = max(template_xmin, 0)
    template_ymin = max(template_ymin, 0)
    template_xmax = min(template_xmax, width)
    template_ymax = min(template_ymax, height)
    # Create the template using the Gaussian blurred image with dimensions defined above.
    template = blurred_img1[template_ymin:template_ymax, template_xmin:template_xmax] 
    
    # Define a search neighbourhood in Image 2. 
    y_min, y_max = y - patch_size2//2, y + patch_size2//2
    x_min, x_max = x - patch_size2//2, x + patch_size2//2
    # Ensure search area is within valid range 
    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(x_max, width)
    y_max = min(y_max, height)
    # Define the search area using the Guassin blurred image with dimensions defined above. 
    search_area = blurred_img2[y_min:y_max, x_min:x_max]
   
    # Perform template matching 
    result = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Apply the threshold 
    if max_val > 0.9:
        # If the value is > 0.9, add the x,y coordinates of the detected corner, the x,y coordinates of the matched conrer and max_val 
        # to the matches list. 
        feature_x = x_min + max_loc[0] # Compute the x coordinate of the corner with respect to the image.  
        feature_y = y_min + max_loc[1] # Compute the y coordinate of the corner with respect to the image. 
    
        matches.append(((x, y),(feature_x, feature_y), max_val))

# endregion ---------------------------------------------------------------------------------------------------------------------------

# region Output Generation ------------------------------------------------------------------------------------------------------------
rows = [] # data to write to csv file 

# Visualization 
for match in matches:
    (img1_x, img1_y), (img2_x, img2_y), corr = match
    
    # Draw a line between the old and new positions on Image 1
    image1 = cv2.circle(image1, (img1_x, img1_y ), 2, (0, 255, 0), -1 ) # green circles (old positions)
    image1 = cv2.line(image1,(img1_x, img1_y),(img2_x, img2_y), (255, 0, 0), 1) # blue lines 
    image1 = cv2.circle(image1, (img2_x, img2_y), 2, (255, 0, 255), -1) # magenta circles (new positions)

    # Append x,y coordinates of the features to the rows list
    row = [img1_x, img1_y, img2_x, img2_y]
    rows.append(row)

# Save coordinates to a csv file 
header = ['image1_x', 'image1_y', 'image2_x', 'image2_y']
with open('feature_coordinates.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    writer.writerows(rows)

# Save the image
cv2.imwrite('image1_results.jpg', image1)

# endregion ---------------------------------------------------------------------------------------------------------------------------
