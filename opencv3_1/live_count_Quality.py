from ultralytics import YOLO
import cv2
import math
import csv

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

model = YOLO('opencv3\opencv3_1\Quality.pt')
classNames = ['O','X']

# CSV 파일을 열고 준비
csv_file = open('opencv3\opencv3_1\detected_objects.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Class'])  # CSV 파일에 헤더 추가

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    object_count = 0  # 물체 카운트 초기화

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            confidence = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])

            cls_name = classNames[min(cls, len(classNames)-1)]

            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            thickness = 2

            cv2.putText(img, cls_name, org, font, fontScale, (255, 0, 0), thickness)

            if cls_name == 'O':
                # 초록색 물체의 클래스를 CSV 파일에 저장
                csv_writer.writerow([cls_name])

            object_count += 1  # 물체 카운트 증가

    # 탐지된 물체의 수를 화면에 표시
    cv2.putText(img, f"Objects: {object_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# CSV 파일 닫기
csv_file.close()
