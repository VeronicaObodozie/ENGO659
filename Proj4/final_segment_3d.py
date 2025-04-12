"""
ENGO659: Digial Imaging and Applications
Project 4: MRI Imaging of Stroke

Written by: Wooju Chung, Veronica Obodozie
Date: April 12, 2025

This script performs lesion and brain segmentation from a series of DICOM-format MRI images. 
It also calculates the area and volume of the segmented regions in both pixel and real-world units, 
and generates a 3D visualization of the detected lesion.
The input data consists of 2D axial DICOM slices of the human brain from a patient diagnosed with ischemic stroke.

PRE-PROCESSING
    1) Load DICOM images and convert pixel arrays to numpy.
    2) Normalize pixel intensity values to the range of 0–255 for lesion segmentation, and 0–1 for brain segmentation.
    3) Apply a bilateral filter to reduce noise while preserving edges in MRI images.

LESION SEGMENTATION
    1) Automatically select a seed point by finding the brightest pixel.
    2) Apply the Region Growing algorithm to segment the lesion area.
    3) Apply morphological closing to refine the lesion mask by removing small holes.
    4) Store each lesion mask in a list for later use in 3D modeling.
    5) Calculate the area and volume of each segmented lesion slice using pixel counts and image spacing information.

3D LESION VISUALIZATION
    1) Stack the segmented lesion masks into a array.
    2) Extract the voxel positions of the lesion.
    3) Render a 3D scatter plot that visualizes the shape and distribution of the lesion.

BRAIN SEGMENTATION
    1) Apply the Morphological Chan-Vese algorithm to each normalized brain slice.
    2) Invert the binary mask if the segmented area exceeds 50% of the image to avoid misclassifying background as brain tissue.
    3) Use morphological opening to remove small artifacts and improve mask quality.
    4) Compute brain area and volume for each slice and accumulate totals.

OUTPUT
    1) 2D lesion and brain masks are optionally saved as binary PNG images.
    2) Printed output includes per-slice and total area/volume statistics.
    3) 3D lesion volume is visualized using a scatter plot of voxel locations.

REFERENCES:
    scikit-image: Region Growing - https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.flood
    scikit-image: Chan-Vese Active Contour - https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.chan_vese
    OpenCV: Bilateral Filter - https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed
    scikit-image: closing - https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.closing
    scikit-image: opening - https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.opening
"""

import os
import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.segmentation import flood, morphological_chan_vese
from skimage.morphology import closing, opening, disk

# --------------------------------------------------------
# Step 1: Lesion Segmentation and Volume Estimation
# --------------------------------------------------------

# Spatial resolution of MRI images (in mm)
pixel_spacing_x, pixel_spacing_y = 0.9375, 0.9375
depth = 3.0

# Load DICOM files
dicom_folder = "/project_4_MR_Stroke/Patient 201/MRI/"
dicom_lesion_files = sorted([os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder) if f.endswith(".dcm")])[25:35]  # for lesion detection
dicom_brain_files = sorted([os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder) if f.endswith(".dcm")])[11:-4]  # for brain segmentation

lesion_masks = []  # store lesion masks for 3D visualization
total_lesion_pixels = 0  # sum of lesion pixels
total_brain_pixels = 0  # sum of brain pixels

# Detect lesions and compute their area and volume
for dicom_file in dicom_lesion_files:
    ds = pydicom.dcmread(dicom_file)
    image = ds.pixel_array.astype(np.float32)

    # Normalize image to range 0–255
    name = os.path.splitext(os.path.basename(dicom_file))[0]
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255

    image = image.astype(np.uint8)

    # Denoise image using bilateral filter
    image_filtered = cv2.bilateralFilter(image, 9, 75, 75)

    # Apply region growing using brightest pixel as seed    
    seed = np.unravel_index(np.argmax(image_filtered), image_filtered.shape)
    segmented_lesion = flood(image_filtered, seed, tolerance=30)
    # remove small noise
    segmented_lesion = closing(segmented_lesion, disk(2))

    # Save mask for 3D visualization
    lesion_masks.append(segmented_lesion.astype(np.uint8))

    # Calculate lesion area and volume
    lesion_area_pixels = np.sum(segmented_lesion)
    lesion_area_mm2 = lesion_area_pixels * pixel_spacing_x * pixel_spacing_y
    lesion_volume_mm3 = lesion_area_mm2 * depth

    total_lesion_pixels += lesion_area_pixels

    # Save binary mask as image
    cv2.imwrite(f'./result/segmented_mask_lesion_{name}.png', segmented_lesion.astype(np.uint8) * 255)

    # Visualize result
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title(f"Original Image [{name}]")
    axes[1].imshow(segmented_lesion, cmap='gray')
    axes[1].set_title("Region Growing (Lesion Segmentation)")
    plt.tight_layout()
    plt.show()
    plt.savefig(f'./result/chart_lesion_{name}.png')
    plt.close(fig)

    print(f"[{name}] Lesion Area: {lesion_area_pixels} pixels, {lesion_area_mm2:.2f} mm², Volume: {lesion_volume_mm3:.2f} mm³")

# Print total lesion area and volume
total_lesion_area_mm2 = total_lesion_pixels * pixel_spacing_x * pixel_spacing_y
total_lesion_volume_mm3 = total_lesion_area_mm2 * depth
print(f"Total Lesion Area: {total_lesion_pixels} pixels, {total_lesion_area_mm2:.2f} mm², Volume: {total_lesion_volume_mm3:.2f} mm³")


# --------------------------------------------------------
# Step 2: 3D Visualization of the Lesion Mask
# --------------------------------------------------------
# Visualize lesion volume in 3D
lesion_masks = np.stack(lesion_masks, axis=0)
lesion_masks = (lesion_masks > 0).astype(np.uint8)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

z, y, x = np.where(lesion_masks == 1)

# 3D scatter plot of lesion voxels
ax.scatter(-y, -x, z, c='red', marker=',', alpha=0.5)

ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Slice")
ax.set_title("Lesion 3D Visualization")

plt.savefig('./result/Lesion_3D_Visualization.png')
plt.show()

# --------------------------------------------------------
# Step 3: Brain Segmentation and Volume Estimation
# --------------------------------------------------------

for dicom_file in dicom_brain_files:
    ds = pydicom.dcmread(dicom_file)
    image = ds.pixel_array.astype(np.float32)

    # Normalize image
    name = os.path.splitext(os.path.basename(dicom_file))[0]
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    # Segment brain using morphological Chan-Vese active contour
    segmented_brain = morphological_chan_vese(image, num_iter=300, smoothing=1, lambda1=0.8, lambda2=1.2)

    # Flip mask if background dominates
    if np.sum(segmented_brain) / segmented_brain.size > 0.5:  
        segmented_brain = 1 - segmented_brain  

    # Remove small objects
    segmented_brain = opening(segmented_brain, disk(4))

    # Save brain mask
    cv2.imwrite(f'./result/segmented_mask_brain_{name}.png', segmented_brain.astype(np.uint8) * 255)

    # Calculate brain area and volume
    brain_area_pixels = np.sum(segmented_brain)  
    brain_area_mm2 = brain_area_pixels * pixel_spacing_x * pixel_spacing_y  
    brain_volume_mm3 = brain_area_mm2 * depth  
    total_brain_pixels += brain_area_pixels  

    print(f"[{name}] Brain Area: {brain_area_pixels} pixels, {brain_area_mm2:.2f} mm², Volume: {brain_volume_mm3:.2f} mm³")

# Print total brain area and volume
total_brain_area_mm2 = total_brain_pixels * pixel_spacing_x * pixel_spacing_y
total_brain_volume_mm3 = total_brain_area_mm2 * depth
print(f"Total Brain Area: {total_brain_pixels} pixels, {total_brain_area_mm2:.2f} mm², Volume: {total_brain_volume_mm3:.2f} mm³")
