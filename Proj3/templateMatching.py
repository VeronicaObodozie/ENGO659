import cv2
import numpy as np
import csv

# Convert binary image and save as jpg
with open('./TemplateMatching/2007135.im', 'rb') as f:
    image_data1 = np.fromfile(f, dtype=np.uint8)

image1_b = image_data1.reshape((960, 720, 1), order='F')
cv2.imwrite('./TemplateMatching/2007135_out.jpg', image1_b)

with open('./TemplateMatching/2007136.im', 'rb') as f:
    image_data2 = np.fromfile(f, dtype=np.uint8)

image2_b = image_data2.reshape((960, 720, 1), order='F')
cv2.imwrite('./TemplateMatching/2007136_out.jpg', image2_b)

image1 = cv2.imread('./TemplateMatching/2007135_out.jpg')
image2 = cv2.imread('./TemplateMatching/2007136_out.jpg')

# Convert image to grayscale
image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Gaussian Blurring 
blurred_img1 = cv2.GaussianBlur(image1_gray, (5, 5), 0)
blurred_img2 = cv2.GaussianBlur(image2_gray, (5, 5), 0)

# region Step 1: Corner Detection 

# Create a blank image (for mask creation)
blank_image = np.zeros((960, 720), dtype=np.uint8)

# Create mask that defines the region of interest (ROI) where corners can be detected 
height, width = 960, 720
mid_height, mid_width = height//2, width//2

# Tile 1 (Top-Left)
mask1 = np.zeros_like(blank_image) # all pixels are 0 (black)
mask1[0:mid_height, 0:mid_width] = 255 # Set the rectangular region pixels to 255 (white)
corners1 = cv2.goodFeaturesToTrack(image1_gray, maxCorners=70, qualityLevel=0.1, minDistance=7, useHarrisDetector=False, mask=mask1) 

# Tile 2 (Top-Right)
mask2 = np.zeros_like(blank_image)
mask2[0:mid_height, mid_width:width] = 255
corners2 = cv2.goodFeaturesToTrack(image1_gray, maxCorners=350, qualityLevel=0.1, minDistance=7, useHarrisDetector=True, mask=mask2) 

# Tile 3 (Bottom-Left)
mask3 = np.zeros_like(blank_image)
mask3[mid_height:height, 0:mid_width] = 255
corners3 = cv2.goodFeaturesToTrack(image1_gray, maxCorners=70, qualityLevel=0.1, minDistance=7, useHarrisDetector=False, mask=mask3) 

# Tile 4 (Bottom-Right)
mask4 = np.zeros_like(blank_image)
mask4[mid_height:height, mid_width:width] = 255
corners4 = cv2.goodFeaturesToTrack(image1_gray, maxCorners=20, qualityLevel=0.1, minDistance=7, useHarrisDetector=False, mask=mask4) 

# Combine the corners from the four ROIs
corners = np.vstack((corners1, corners2, corners3, corners4))

# endregion 

# Visualize corners on Image1
# corners = np.float32(corners)
# image1_feature_coordinates = []
# for corner in corners:
#     x, y = corner.ravel()
#     #image1_feature_coordinates.append((int(x), int(y)))
#     cv2.circle(image1, (int(x), int(y)), 3, (0, 255, 0), -1) 


# region Step 2 - 4: Feature Matching 
patch_size1 = 6
patch_size2 = 24
matches = []
image1_features = []
for corner in corners:
    x, y = corner.ravel()
    x, y = int(x), int(y)

    #print(f"Corner {corner}")
    #print(f"feature x coordinate: {x}, y coordinate: {y}")

    # Step 2: Create template using feature location from corners detected in Image 1 
    template_ymin, template_ymax = y-patch_size1//2, y+patch_size1//2
    template_xmin, template_xmax = x-patch_size1//2, x+patch_size1//2

    # Ensure template is within valid range
    template_xmin = max(template_xmin, 0)
    template_ymin = max(template_ymin, 0)
    template_xmax = min(template_xmax, width)
    template_ymax = min(template_ymax, height)

    template = blurred_img1[template_ymin:template_ymax, template_xmin:template_xmax] 
    
    #print(f"template x coordinate: {( x-patch_size//2 + x+patch_size//2)//2}")

    # Step 3: Define a search neighbourhood in Image 2 and perform template matching between the template and search area
    y_min, y_max = y - patch_size2//2, y + patch_size2//2
    x_min, x_max = x - patch_size2//2, x + patch_size2//2
    #print(f"search_area before checking: x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max: {y_max}")
    
    # Ensure search area is within valid range 
    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(x_max, width)
    y_max = min(y_max, height)

    search_area = blurred_img2[y_min:y_max, x_min:x_max]
    #print(f"search_area after checking: x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max: {y_max}")
    
    # Step 4: Perform template matching 
    result = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    #print(f"min_val: {min_val}, max_val: {max_val}, min_loc: {min_loc}, max_loc: {max_loc}")

    feature_x = x_min + max_loc[0]
    feature_y = y_min + max_loc[1]
    #print(f"feature_x: {feature_x}, feature_y: {feature_y}")

    # Apply threshold: max_val > 0.9
    if max_val > 0.9:
        matches.append(((x, y),(feature_x, feature_y), max_val))

# endregion

# region visualize
rows = []
for match in matches:
    (img1_x, img1_y), (img2_x, img2_y), corr = match
    #print(f"corrleation: {corr}")
    #print(f"image1_x:{img1_x}, image1_y:{img1_y}, image2_x:{img2_x}, image2_y:{img2_y}")

    # Draw a line between the old and new positions 
    image1 = cv2.circle(image1, (img1_x, img1_y ), 2, (0, 255, 0), -1 ) # green circles (old positions)
    image1 = cv2.line(image1, (img2_x, img2_y), (img1_x, img1_y), (255, 0, 0),1 )
    image1 = cv2.circle(image1, (img2_x, img2_y), 2, (255, 0, 255), -1) # magenta circles (new positions)

    row = [img1_x, img1_y, img2_x, img2_y]
    rows.append(row)

# save coordinates to a csv file 
header = ['image1_x', 'image1_y', 'image2_x', 'image2_y']
with open('./TemplateMatching/feature_coordinates.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    writer.writerows(rows)

#print("after threshold: ", len(matches))
cv2.imwrite('./TemplateMatching/image1_results.jpg', image1)
# endregion 
