import cv2
import numpy as np
import csv

# Load two images 
image1 = cv2.imread('2007135_out.jpg')
image2 = cv2.imread('2007136_out.jpg')

# Convert image to grayscale
image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Create a blank image (for mask creation)
blank_image = np.zeros((720, 960), dtype=np.uint8)

# Create mask that defines the region of interest (ROI) where corners can be detected 
height, width = 720, 960
mid_height, mid_width = height//2, width//2

# Tile 1 (Top-Left)
mask1 = np.zeros_like(blank_image) # all pixels are 0 (black)
mask1[0:mid_height, 0:mid_width] = 255 # Set the rectangular region pixels to 255 (white)
corners1 = cv2.goodFeaturesToTrack(image1_gray, maxCorners=20, qualityLevel=0.3, minDistance=7, useHarrisDetector=False, mask=mask1) 

# Tile 2 (Top-Right)
mask2 = np.zeros_like(blank_image)
mask2[0:mid_height, mid_width:width] = 255
corners2 = cv2.goodFeaturesToTrack(image1_gray, maxCorners=20, qualityLevel=0.3, minDistance=7, useHarrisDetector=False, mask=mask2) 

# Tile 3 (Bottom-Left)
mask3 = np.zeros_like(blank_image)
mask3[mid_height:height, 0:mid_width] = 255
corners3 = cv2.goodFeaturesToTrack(image1_gray, maxCorners=60, qualityLevel=0.3, minDistance=7, useHarrisDetector=True, mask=mask3) 

# Tile 4 (Bottom-Right)
mask4 = np.zeros_like(blank_image)
mask4[mid_height:height, mid_width:width] = 255
corners4 = cv2.goodFeaturesToTrack(image1_gray, maxCorners=20, qualityLevel=0.3, minDistance=7, useHarrisDetector=False, mask=mask4) 

# Combine the corners from the four ROIs
corners = np.vstack((corners1, corners2, corners3, corners4))

# Visualize corners on Image1
corners = np.float32(corners)
image1_feature_coordinates = []
for corner in corners:
    x, y = corner.ravel()
    image1_feature_coordinates.append((int(x), int(y)))
    cv2.circle(image1, (int(x), int(y)), 3, (0, 255, 0), -1) 

#cv2.imwrite('image1_corners.jpg', image1)

# Define Lucas-Kanade optical flow parameters
lk_params = dict(
    winSize = (30, 30), # Defaut size
    maxLevel = 3, # Default value (number of pyramid levels)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03), # Termination criteria
    flags = 0, 
    minEigThreshold = 1e-4  # Minimum eigenvalue threshold
)
# Track the corenrs from the first image to the second image using optical flow 
corners2, status, error = cv2.calcOpticalFlowPyrLK(image1_gray, image2_gray, corners, None, **lk_params)
    
# Visualize corners on Image2
corners2 = np.float32(corners2)
image2_feature_coordinates = []
for corner in corners2:
    x, y = corner.ravel()
    image2_feature_coordinates.append((int(x), int(y)))
    cv2.circle(image2, (int(x), int(y)), 3, (255, 0, 255), -1) 

# Visualize vectors between two corners 
# for i, (new,old) in enumerate(zip(corners2, corners)):
#     a, b = new.ravel() 
#     c, d = old.ravel()

#     # Draw a line between the old and new positions 
#     image2 = cv2.circle(image2, (int(c), int(d)), 2, (0, 255, 0), -1 ) # green circles 
#     image2 = cv2.line(image2, (int(a), int(b)), (int(c), int(d)), (255, 0, 0), 1) # (0, 255, 0) - blue lines
#     image2 = cv2.circle(image2, (int(a), int(b)), 2, (255, 0, 255), -1) # magenta circles 

# cv2.imwrite('image2_results.jpg', image2)

# Save coordinates of features in image1 and image2 to a csv file 
image1_x, image1_y = zip(*image1_feature_coordinates)
image2_x, image2_y = zip(*image2_feature_coordinates)

rows = zip(image1_x, image1_y, image2_x, image2_y)
header = ['image1_x', 'image1_y', 'image2_x', 'image2_y']

# Write data to csv
with open('feature_coordinates.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    writer.writerows(rows)