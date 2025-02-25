

# # alternative code for getting the number
# for i, c in enumerate(cnt):
#     # 윤곽선
#     cv2.drawContours(rgb, [c], -1, (0, 255, 0), 2)  

#     # 좌표계산해 숫자넣기
#     M = cv2.moments(c)
#     if M["m00"] != 0:  # 면적이 0
#         cx = int(M["m10"] / M["m00"])
#         cy = int(M["m01"] / M["m00"])
#     else:  # 면적이 0인 객체에 대한 임시 좌표
#         cx, cy = c[0][0]  # 객체의 첫째 점
    
#     if i == 48 or i == 51:  #4 9번과 52번 인덱스 48, 51
#         print(f"Object {i+1} - Centroid: ({cx}, {cy})")
#     # cv2.circle(rgb, (cx, cy), 5, (0, 0, 255), -1)  # 원그리기
#     # 번호 표시
#     cv2.putText(rgb, str(i+1), (cx-30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
# plt.figure(5)
# rgb = cv2.resize(rgb, None, fx=1.5, fy=1.5)  # 크기 1.5배 확대
# plt.title('Detected: '+ str(len(cnt))+' Tents in Region')
# plt.imshow(rgb)
# plt.show()
# print("Number of tents in region of interest: ", len(cnt))