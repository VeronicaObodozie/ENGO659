# 뇌 세그멘트 퓨어 active contour chan vese
from skimage.segmentation import morphological_chan_vese
import numpy as np
from skimage.measure import find_contours
from skimage.morphology import closing, opening, disk
import os
import pydicom
import numpy as np
import cv2
from skimage.filters import sobel
from skimage.segmentation import flood
import matplotlib.pyplot as plt

# DICOM 파일 로드

dicom_folder = "/Users/wj/Downloads/ENGO 659/pj4/project_4_MR_Stroke/Patient 201/MRI/"
dicom_files = sorted([os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder) if f.endswith(".dcm")])[11:25]  # 뇌 앞부분
# dicom_files = sorted([os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder) if f.endswith(".dcm")])[35:90]  # 뇌 뒷부분

#  여러 장의 이미지 처리
for idx, dicom_file in enumerate(dicom_files):

    ds = pydicom.dcmread(dicom_file)
    image = ds.pixel_array.astype(np.float32)

    #  이미지 전처리 (정규화)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    name = os.path.basename(dicom_file)
    # 확장자 제거 (선택 사항)
    name = os.path.splitext(name)[0]

    # Chan-Vese 모델 적용 
    segmented = morphological_chan_vese(
        image, num_iter=300,  # 반복 횟수 증가 시 더 정확한 경계 탐색 가능
        smoothing=1,  # 스무딩 정도 (경계를 더 부드럽게)
        lambda1=0.8, lambda2=1.2  # 내부/외부 강도 비율 조정
    )
    # 세그멘테이션 마스크 비율 확인
    if np.sum(segmented) / segmented.size > 0.5:  
        segmented = 1 - segmented  # 값 반전
    
    #작은거 없애게 opening
    segmented = opening(segmented, disk(5))  # 작은 객체 제거

    # cv2.imwrite(f'./result_brain/segmented_mask_{name}.png', segmented.astype(np.uint8) * 255)  # 0과 1의 값이므로 255로 스케일링하여 저장

    # 세그멘테이션된 영역의 경계 추출
    contours = find_contours(segmented, 0.5)

    # 새로만든거
    #  결과 시각화 (원본, 세그멘테이션 마스크, 컨투어 오버레이)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 원본 이미지
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title(f"Original Image {name}")
    # axes[0].axis("off")

    # 세그멘테이션 마스크
    axes[1].imshow(segmented, cmap='gray')
    axes[1].set_title("Segmented Mask")
    axes[1].axis("off")

    # 원본 + 컨투어 오버레이
    axes[2].imshow(image, cmap='gray')
    for contour in contours:
        axes[2].plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')
    axes[2].set_title("Segmented Contour Overlay")
    axes[2].axis("off")

    object_size = np.sum(segmented)  # 객체 부분의 픽셀 수

    # 객체의 면적을 출력
    print(f"Object Size of {name} (in pixels): {object_size}")

    plt.show()