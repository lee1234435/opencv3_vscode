import cv2
from ultralytics import YOLO

model = YOLO("D:/ROS&C&PYTHONCODE/코드관련모든파일/opencv3/opencv3_1/best_best.pt") # 코랩에서 학습 후 모델 (코랩해서 학습해서 신호등 초록, 빨강색 구분 모델용)
# model = YOLO("D:/ROS&C&PYTHONCODE/코드관련모든파일/opencv3/opencv3_1/best2.pt") # 설희가 만든 모델

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    results = model(img)
    annotated_frame = results[0].plot()
    cv2.imshow("webcam",annotated_frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# model = YOLO("yolov8n.pt") #yolo 커스텀 전 기본 모델
# model.train(data='data.yaml', imgsz=640,workers=4, momentum=0.9,epochs=10,batch=2) #모델 학습
# model = YOLO('runs\detect\train_pen\weights\best.pt') # 학습 후 모델 (pen)
# model = YOLO('./runs/detect/train_coin/weights/best.pt') # 학습 후 모델 (coin)
# model = YOLO('./runs/detect/train_traffic_1/weights/best.pt') # 학습 후 모델 (traffic_1)
# model = YOLO("D:/ROS&C&PYTHONCODE/코드관련모든파일/opencv3/opencv3_1/best_best.pt") # 블록 4가지 색깔 학습한 모델


