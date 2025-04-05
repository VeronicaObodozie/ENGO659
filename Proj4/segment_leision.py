#  필터로 노이즈 + 에지디텍션 +히스토그램기준 시드포인트 자동으로 찾기 +region growing
import os
import pydicom
import numpy as np
import cv2
from skimage.filters import sobel
from skimage.segmentation import flood
import matplotlib.pyplot as plt
from skimage.morphology import closing, opening, disk



#  DICOM 파일 로드

dicom_folder = "/Users/wj/Downloads/ENGO 659/pj4/project_4_MR_Stroke/Patient 201/MRI/"
dicom_files = sorted([os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder) if f.endswith(".dcm")])[25:35]  # 10장만 선택

#2️⃣ 여러 장의 이미지 처리
for idx, dicom_file in enumerate(dicom_files):

    ds = pydicom.dcmread(dicom_file)
    image = ds.pixel_array.astype(np.float32)

    # 파일 이름 추출
    name = os.path.basename(dicom_file)
    # 확장자 제거 (선택 사항)
    name = os.path.splitext(name)[0]

    # 정규화 (0~255 범위로 조정)
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    image = image.astype(np.uint8)

    # 가우시안 필터 적용 (커널 크기 5x5, 표준편차 1)
    # image = cv2.GaussianBlur(image, (5, 5), 1)
    image = cv2.bilateralFilter(image, 9, 75, 75)  # Bilateral Filter

    # Edge Detection (Sobel 필터)
    edges = sobel(image)

    # Edge가 강한 부분(경계선)에는 Region Growing이 확장되지 않도록 마스크 적용
    mask = edges < np.percentile(edges, 90)  # 상위 10% edge는 확장 금지

# 수기 세팅
    # seed = (image.shape[0] // 2, image.shape[1] // 2)
    # seed = (150,120)

    # 히스토그램 분석 후 가장 밝은 부분 찾기
    seed = np.unravel_index(np.argmax(image), image.shape)

    # Region Growing 수행
    segmented_with_edges = flood(image, seed, tolerance=30)

     # closing
    closed_mask = closing(segmented_with_edges, disk(2))  # 3은 구조 요소의 크기 (원하는 크기 조정 가능)

    # 결과 시각화

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 3, 1)
    # image_with_seed = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # 컬러로 변환
    # cv2.circle(image_with_seed, seed, 5, (0, 0, 255), -1)  # 씨드를 빨간색으로 표시
    # plt.imshow(image_with_seed, cmap='gray')
    plt.imshow(image, cmap='gray')
    plt.title(f"filtered Image [{name}]")
    # plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(edges, cmap='gray')
    plt.title("Edge Detection (Sobel)")
    plt.axis("off")

    # 최종 마스크를 컬러 이미지로 변환하여 시드 위치를 표시
    # mask_with_seed = np.uint8(segmented_with_edges) * 255  # 이진 마스크 (0, 255)
    # mask_with_seed = cv2.cvtColor(mask_with_seed, cv2.COLOR_GRAY2BGR)  # 컬러 변환
    # cv2.circle(mask_with_seed, seed, 5, (0, 0, 255), -1)  # 씨드 빨간색 점 추가

    plt.subplot(1, 3, 3)
    plt.imshow(segmented_with_edges, cmap='gray')

    plt.imshow(closed_mask, cmap='gray')

    # plt.imshow(mask_with_seed, cmap='gray') # 시드포인트 위치 보여주기
    plt.title("Region Growing with Edge Constraint")
    # plt.axis("off")

    # save the masks
    # cv2.imwrite(f'./result/segmented_mask_{name}.png', segmented_with_edges.astype(np.uint8) * 255)  # 0과 1의 값이므로 255로 스케일링하여 저장

    plt.show()
        # 객체(병변)의 크기 계산
    object_size = np.sum(closed_mask)  # 객체 부분의 픽셀 수

    # 객체의 면적을 출력
    print(f"Object Size of {name} (in pixels): {object_size}")