# QR코드 입력
import cv2

def decode_qrcode(cap):
    if not cap.isOpened():
        print("Camera open failed!")
        return

    detector = cv2.QRCodeDetector()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Frame load failed!")
            break

        info = detector.detectAndDecodeMulti(frame)

        if info[0]:  # QR 코드가 검출되었는지 확인
            print("Detected QR code:", info[1])

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) == ord('q'):  # 'q' 키를 누르면 종료
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    cap = cv2.VideoCapture(0)

    decode_qrcode(cap)

if __name__ == "__main__":
    main()
