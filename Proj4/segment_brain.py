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

# 1️⃣ DICOM 파일 로드

dicom_folder = "/Users/wj/Downloads/ENGO 659/pj4/project_4_MR_Stroke/Patient 201/MRI/"
dicom_files = sorted([os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder) if f.endswith(".dcm")])[11:25]  # 뇌 앞부분
# dicom_files = sorted([os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder) if f.endswith(".dcm")])[35:90]  # 뇌 뒷부분

# 2️⃣ 여러 장의 이미지 처리
for idx, dicom_file in enumerate(dicom_files):

    ds = pydicom.dcmread(dicom_file)
    image = ds.pixel_array.astype(np.float32)

    # 2️⃣ 이미지 전처리 (정규화)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    # Chan-Vese 모델 적용 
    segmented = morphological_chan_vese(
        image, num_iter=300,  # 반복 횟수 증가 시 더 정확한 경계 탐색 가능
        smoothing=1,  # 스무딩 정도 (경계를 더 부드럽게)
        lambda1=1, lambda2=1  # 내부/외부 강도 비율 조정
    )
    # 세그멘테이션 결과가 1인 부분을 붉은색(RGB)으로 덧씌움
    overlay = np.zeros((*segmented.shape, 3), dtype=np.uint8)
    overlay[segmented == 1] = [255, 0, 0]  # 빨간색(RGB)

    # # 결과 시각화
    # plt.figure(figsize=(8, 6))
    # plt.imshow(segmented, cmap='gray')
    # plt.title("Chan-Vese Active Contour Result")
    # plt.axis("off")
    # plt.show()

    # # 원본 이미지와 오버레이 결합
    # plt.figure(figsize=(8, 6))
    # plt.imshow(image, cmap='gray')  # 원본 이미지
    # plt.imshow(overlay, alpha=0.5)  # 투명도 0.5로 마스크 오버레이
    # plt.title("Segmented Region Overlay")
    # plt.axis("off")
    # plt.show()
    
    #작은거 없애게 opening
    segmented = opening(segmented, disk(5))  # 작은 객체 제거

    # 세그멘테이션된 영역의 경계 추출
    contours = find_contours(segmented, 0.5)

    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap='gray')

    # 경계를 빨간색으로 표시
    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')

    plt.title("Segmented Contour Overlay")
    plt.axis("off")
    plt.show()