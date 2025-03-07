import numpy as np
import cv2
import os


def process_binary_file(filepath):

    with open(filepath, 'rb') as f:
        image_data = np.fromfile(f, dtype=np.uint8)

    width, height, channel = 960, 720, 1

    image = image_data.reshape((height, width, channel))
    
    return image 


def process_im_files_in_folder(folder_path):
    
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        if file.endswith('.im'):
            print(f"Processing file: {file} in {file_path}" )
            image = process_binary_file(file_path)

            # Save image as .jpg
            cv2.imwrite(str(file).replace('.im', '_out.jpg'), image)


if __name__ == "__main__":

    #folder_path = './project_3_SeaIceMotion/'
    #process_im_files_in_folder(folder_path)

    # Load two images 
    image1 = cv2.imread('2007135_out.jpg')
    image2 = cv2.imread('2007136_out.jpg')

    # Convert image to grayscale
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # Detect good features to track (corners) in image1
    #TODO: Look into manually selecting features to track 
    corners1 = cv2.goodFeaturesToTrack(image1_gray, maxCorners=100, qualityLevel=0.3, minDistance=7) # image1_goodFeatures.jpg   
    #corners1 = cv2.goodFeaturesToTrack(image1_gray, maxCorners=100, qualityLevel=0.1, minDistance=7) # image1_goodFeatures_v2.jpg
    #corners1 = cv2.goodFeaturesToTrack(image1_gray, maxCorners=300, qualityLevel=0.1, minDistance=7) # image1_goodFeatures_v3.jpg
    
    # Convert corners to a float32 type 
    corners1 = np.float32(corners1)

    # Draw circles on the corners detected 
    # for corner in corners1:
    #     x, y = corner.ravel()
    #     cv2.circle(image1, (int(x), int(y)), 3, (0, 255, 0), -1) 

    # cv2.imwrite('image1_goodFeatures_v3.jpg', image1)

    # Track the corenrs from the first image to the second image using optical flow 
    corners2, status, error = cv2.calcOpticalFlowPyrLK(image1_gray, image2_gray, corners1, None)
    print(status) # 1 - successful, 0 - fail

    # Visualize the tracked points
    for i, (new,old) in enumerate(zip(corners2, corners1)):
        a, b = new.ravel()
        c, d = old.ravel()

        # Draw a line between the old and new positions 
        image2 = cv2.line(image2, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        image2 = cv2.circle(image2, (int(a), int(b)), 5, (0, 0, 255), -1)

    cv2.imwrite('image2_result.jpg', image2)