import cv2
import os
import time

def capture_image():
    image_count = 0
    cap = cv2.VideoCapture(0)
    save_directory  = "/img_capture"
    
    os.makedirs(save_directory, exist_ok=True)
    while True:
        ret,frame = cap.read()
        cv2.imshow("webcam",frame)
        
        key = cv2.waitKey(1)
        
        if key == ord('c'):
            file_name = f'{save_directory}/img_{image_count}.jpg'
            cv2.imwrite(file_name,frame)
            print(f"Image Saved. name:{file_name}")
            image_count += 1 
        elif key == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

capture_image()


# import cv2
# # matplotlib.image 를 사용하기 위해선 matplotlib 뿐만 아니라 pillow도 깔아야 한다.
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg


# # 색상 범위 설정 (orange) 잡음)
# lower_orange = (100, 100, 100)
# upper_orange = (200, 200, 255)

# # 색상 범위 설정 (green)
# lower_green = (30, 80, 80)  
# upper_green = (70, 255, 255)

# # 색상 범위 설정 (blue)
# lower_blue = (0, 180, 55)
# upper_blue = (20, 255, 200)

# # 색상 범위 설정 (red)
# lower_red = (0, 0, 0)
# upper_red = (255, 255, 255)

# # 색상 범위 설정 (yellow)
# lower_yellow = (0, 0, 0)
# upper_yellow = (255, 255, 255)

# # # 이미지 파일을 읽어온다
# img_orange = mpimg.imread("img_capture\img_orange.jpg", cv2.IMREAD_COLOR)
# img_blue = mpimg.imread("img_capture\img_blue.jpg", cv2.IMREAD_COLOR)
# img_green = mpimg.imread("img_capture\img_green.jpg", cv2.IMREAD_COLOR)

# img_red = mpimg.imread("img_capture\img_red.jpg", cv2.IMREAD_COLOR)
# img_yellow = mpimg.imread("img_capture\img_yellow.jpg", cv2.IMREAD_COLOR)

# # BGR to HSV 변환
# img_hsv_orange = cv2.cvtColor(img_orange, cv2.COLOR_BGR2HSV)
# img_hsv_blue = cv2.cvtColor(img_blue, cv2.COLOR_BGR2HSV)
# img_hsv_green = cv2.cvtColor(img_green, cv2.COLOR_BGR2HSV)
# img_hsv_red = cv2.cvtColor(img_red, cv2.COLOR_BGR2HSV)
# img_hsv_yellow = cv2.cvtColor(img_yellow, cv2.COLOR_BGR2HSV)

# # 색상 범위를 제한하여 mask 생성
# img_mask_orange = cv2.inRange(img_hsv_orange, lower_orange, upper_orange)

# # 색상 범위를 제한하여 mask 생성
# img_mask_blue = cv2.inRange(img_hsv_blue, lower_blue, upper_blue)

# # 색상 범위를 제한하여 mask 생성
# img_mask_green = cv2.inRange(img_hsv_green, lower_green, upper_green)

# # 색상 범위를 제한하여 mask 생성
# img_mask_red = cv2.inRange(img_hsv_red, lower_red, upper_red)

# # 색상 범위를 제한하여 mask 생성
# img_mask_yellow = cv2.inRange(img_hsv_yellow, lower_yellow, upper_yellow)

# # 원본 이미지를 가지고 Object 추출 이미지로 생성
# img_result_orange = cv2.bitwise_and(img_orange, img_orange, mask=img_mask_orange)

# img_result_blue = cv2.bitwise_and(img_blue, img_blue, mask=img_mask_blue)

# img_result_green = cv2.bitwise_and(img_green, img_green, mask=img_mask_green)

# img_result_red = cv2.bitwise_and(img_red, img_red, mask=img_mask_red)

# img_result_yellow = cv2.bitwise_and(img_yellow, img_yellow, mask=img_mask_yellow)

# 결과 이미지 생성
# imgplot_orange = plt.imshow(img_result_orange)

# imgplot_blue = plt.imshow(img_result_blue)

# imgplot_green = plt.imshow(img_result_green)

# imgplot_red = plt.imshow(img_result_red)

# imgplot_yellow = plt.imshow(img_result_yellow)

# plt.figure(figsize=(10, 5))

# plt.subplot(1, 5, 1)
# plt.imshow(img_result_orange)
# plt.title('Orange')

# plt.subplot(1, 5, 2)
# plt.imshow(img_result_blue)
# plt.title('Blue')

# plt.subplot(1, 5, 3)
# plt.imshow(img_result_green)
# plt.title('Green')

# plt.subplot(1, 5, 4)
# plt.imshow(img_result_red)
# plt.title('Red')

# plt.subplot(1, 5, 5)
# plt.imshow(img_result_yellow)
# plt.title('Yellow')

# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.image as mping


# def RGB2HSV(RGB):
#     # HSV 색상을 얻기 위해서는 array 타입이 float이 되어야 계산할 수 있다
#     RGB_array = np.array(RGB).astype(np.float64)
    
#     # 변환할 HSV 이미지 생성
#     HSV = np.array(RGB).astype(np.float64)
    
#     # RGB 이미지의 width, height 저장
#     width, height = RGB_array.shape[:2]

#     # 이미지 크기만큼 for 루프
#     for i in range(width):
#         for j in range(height):
#             # 공식 따라서 구현
#             var_R = RGB_array[i, j, 0] / 255.0
#             var_G = RGB_array[i, j, 1] / 255.0
#             var_B = RGB_array[i, j, 2] / 255.0

#             C_Min = min(var_R, var_G, var_B)
#             C_Max = max(var_R, var_G, var_B)

#             change = C_Max - C_Min
#             V = C_Max
            
#             if C_Max == 0:
#                 S = 0
#             else:
#                 S = change / C_Max
                
#             if change == 0:
#                 H = 0
#             else:
#                 if var_R == C_Max:
#                     H = 60 * (((var_R - var_B)/change)%6)
#                 elif var_G == C_Max:
#                     H = 60 * (((var_B - var_R)/change)+2)
#                 elif var_B == C_Max:
#                     H = 60 * (((var_R - var_B)/change)+4)
                    
#             HSV[i, j, 0] = H
#             HSV[i, j, 1] = S
#             HSV[i, j, 2] = V     
            
#     return HSV

# def Mask(HSV, color):
#     # 범위값과 비교할 hsv 이미지 생성, 파라미터에 있는 HSV를 그냥 쓰면 원소값이 float이 아닌 int로 나옴
#     hsv = np.array(HSV).astype(np.float64)

#     # HSV 이미지의 width, height 저장
#     width, height = HSV.shape[:2]

#     # 모든 값은 원소 값이 0 인 마스크 행렬 생성
#     mask = np.zeros((width, height))

#     # hsv 값과 범위 비교
#     for i in range(width):
#         for j in range(height):
#             # H, S, V 값이 원하는 범위 안에 들어갈 경우 mask 원소 값을 1로 만든다
#             if hsv[i, j, 0] > lower[color][0] and hsv[i, j, 1] > lower[color][1] and hsv[i, j, 2] > lower[color][2] and hsv[i, j, 0] < upper[color][0] and hsv[i, j, 1] < upper[color][1] and hsv[i, j, 2] < upper[color][2]:
#                 mask[i, j] = 1
                
#     return mask

# def Extraction(image, mask):
#     # Object를 추출할 이미지를 생성
#     result_img = np.array(image)

#     # RGB 이미지의 width, height 저장
#     width, height = image.shape[:2]

#     # for 루프를 돌면서 mask 원소 값이 0인 인덱스는 원본 이미지도 0으로 만들어 준다.
#     for i in range(width):
#         for j in range(height):
#             if(mask[i, j] == 0):
#                 result_img[i, j, 0] = 0
#                 result_img[i, j, 1] = 0
#                 result_img[i, j, 2] = 0
                
#     return result_img


# if __name__ == '__main__':
#     # 마스크 색상 범위에 사용할 딕셔너리 정의
#     upper = {}
#     lower = {}

#     upper['orange'] = [100, 1, 1]
#     upper['blue'] = [300, 1, 1]
#     upper['green'] = [180, 0.7, 0.5]

#     lower['orange'] = [0, 0.7, 0.5]
#     lower['blue'] = [70, 0.7, 0.2]
#     lower['green'] = [101, 0.15, 0]

#     # 이미지 파일을 읽어온다
#     input_image = mping.imread('img_capture\img_3.jpg')

#     # 추출하고 싶은 색상 입력
#     input_color = input("추출하고 싶은 색상을 입력하세요 (orange, blue, green) : ")

#     # RGB to HSV 변환
#     HSV = RGB2HSV(input_image)

#     # HSV 이미지를 가지고 마스크 생성
#     mask = Mask(HSV, input_color)

#     # mask를 가지고 원본이미지를 Object 추출 이미지로 변환
#     result_image = Extraction(input_image, mask)

#     #mping.imsave("result.jpg", result_image)

#     # 이미지 보여주기
#     imgplot = plt.imshow(result_image)

#     plt.show()