import os
import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import sobel
from skimage.segmentation import flood, morphological_chan_vese
from skimage.morphology import closing, opening, disk
# from skimage.measure import find_contours
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D

# MRI 이미지의 실제 공간 해상도 (mm)
pixel_spacing_x, pixel_spacing_y = 0.9375, 0.9375  # 픽셀 간격
depth = 3.0  # 슬라이스 간격 (mm)

# DICOM 파일 불러오기
dicom_folder = "/Users/wj/Downloads/ENGO 659/pj4/project_4_MR_Stroke/Patient 201/MRI/"
dicom_lesion_files = sorted([os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder) if f.endswith(".dcm")])[25:35]  # 병변 검출용
dicom_brain_files = sorted([os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder) if f.endswith(".dcm")])[11:-4]  # 뇌 전체 분할용

lesion_masks = []  # 3D 모델 생성을 위한 병변 마스크 저장 리스트
total_lesion_pixels = 0  # 병변 픽셀 총합
total_brain_pixels = 0  # 뇌 픽셀 총합

#  병변(Lesion) 검출 및 면적/부피 계산
for dicom_file in dicom_lesion_files:
    ds = pydicom.dcmread(dicom_file)
    image = ds.pixel_array.astype(np.float32)

    # 파일 이름 추출 및 정규화 (0~255 범위로 조정)
    name = os.path.splitext(os.path.basename(dicom_file))[0]
    # image = (image - np.min(image)) / (np.max(image)) * 255
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255 #이거 더 나을수도

    image = image.astype(np.uint8)

    #  노이즈 제거 (Bilateral Filter)
    image_filtered = cv2.bilateralFilter(image, 9, 75, 75)

    #  Edge Detection (Sobel 필터)
    edges = sobel(image_filtered)
    mask = edges < np.percentile(edges, 90)  # 상위 10% edge는 확장 금지

    #  Region Growing (히스토그램 분석 후 가장 밝은 부분을 seed로 사용)
    seed = np.unravel_index(np.argmax(image_filtered), image_filtered.shape)
    segmented_lesion = flood(image_filtered, seed, tolerance=30)
    segmented_lesion = closing(segmented_lesion, disk(2))  # 작은 객체 제거

    # 3D 모델 생성을 위해 마스크 저장
    lesion_masks.append(segmented_lesion.astype(np.uint8))

    #  병변 면적 및 부피 계산
    lesion_area_pixels = np.sum(segmented_lesion)  # 픽셀 단위 면적
    lesion_area_mm2 = lesion_area_pixels * pixel_spacing_x * pixel_spacing_y  # mm² 단위 면적
    lesion_volume_mm3 = lesion_area_mm2 * depth  # mm³ 단위 부피

    total_lesion_pixels += lesion_area_pixels  # 전체 픽셀 합산

    #마스크 파일에 저장
    cv2.imwrite(f'./result/segmented_mask_lesion_{name}.png', segmented_lesion.astype(np.uint8) * 255)  # 0과 1의 값이므로 255로 스케일링하여 저장

    # #  병변 결과 시각화
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # axes[0].imshow(image, cmap='gray')
    # axes[0].set_title(f"Original Image [{name}]")

    # axes[1].imshow(edges, cmap='gray')
    # axes[1].set_title("Edge Detection (Sobel)")

    # axes[2].imshow(segmented_lesion, cmap='gray')
    # axes[2].set_title("Region Growing (Lesion Segmentation)")

    # plt.tight_layout()
    # plt.savefig(f'./result/chart_lesion_{name}.png')
    # plt.close(fig)

    print(f"[{name}] Lesion Area: {lesion_area_pixels} pixels, {lesion_area_mm2:.2f} mm², Volume: {lesion_volume_mm3:.2f} mm³")
# plt.savefig('./Lesion.png')
# plt.show()

#  전체 병변의 면적 및 부피 출력
total_lesion_area_mm2 = total_lesion_pixels * pixel_spacing_x * pixel_spacing_y
total_lesion_volume_mm3 = total_lesion_area_mm2 * depth
print(f"Total Lesion Area: {total_lesion_pixels} pixels, {total_lesion_area_mm2:.2f} mm², Volume: {total_lesion_volume_mm3:.2f} mm³")

#  병변(Lesion) 3D 모델 생성
lesion_masks = np.stack(lesion_masks, axis=0)  # 마스크 스택 쌓기 (10, 256, 256)
lesion_masks = (lesion_masks > 0).astype(np.uint8)  # 이진화 (0, 1)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# x, y, z 좌표 생성
z, y, x = np.where(lesion_masks == 1)  # 마스크가 1인 부분만 플로팅

# 점 그래프 형태로 시각화
ax.scatter(-y, -x, z, c='red', marker=',', alpha=0.5)

ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Slice")
ax.set_title("Lesion 3D Visualization")

plt.savefig('./result/Lesion_3D_Visualization.png')
# plt.show()

#  뇌(Brain) 분할 및 면적/부피 계산
for dicom_file in dicom_brain_files:
    ds = pydicom.dcmread(dicom_file)
    image = ds.pixel_array.astype(np.float32)

    # 파일 이름 추출 및 정규화 (0~1 범위로 조정)
    name = os.path.splitext(os.path.basename(dicom_file))[0]
    # image = (image - np.min(image)) / (np.max(image))
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    #  Chan-Vese Active Contour 적용
    segmented_brain = morphological_chan_vese(image, num_iter=300, smoothing=1, lambda1=0.8, lambda2=1.2)

    #  마스크 반전
    if np.sum(segmented_brain) / segmented_brain.size > 0.5:  
        segmented_brain = 1 - segmented_brain  

    #  작은 객체 제거
    segmented_brain = opening(segmented_brain, disk(4))

    #  세그멘테이션 마스크 저장
    cv2.imwrite(f'./result/segmented_mask_brain_{name}.png', segmented_brain.astype(np.uint8) * 255)  # 0과 1의 값이므로 255로 스케일링하여 저장

    #  뇌 면적 및 부피 계산
    brain_area_pixels = np.sum(segmented_brain)  
    brain_area_mm2 = brain_area_pixels * pixel_spacing_x * pixel_spacing_y  
    brain_volume_mm3 = brain_area_mm2 * depth  

    total_brain_pixels += brain_area_pixels  

    print(f"[{name}] Brain Area: {brain_area_pixels} pixels, {brain_area_mm2:.2f} mm², Volume: {brain_volume_mm3:.2f} mm³")

#  전체 뇌의 면적 및 부피 출력
total_brain_area_mm2 = total_brain_pixels * pixel_spacing_x * pixel_spacing_y
total_brain_volume_mm3 = total_brain_area_mm2 * depth
print(f"Total Brain Area: {total_brain_pixels} pixels, {total_brain_area_mm2:.2f} mm², Volume: {total_brain_volume_mm3:.2f} mm³")
