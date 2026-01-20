from ultralytics import YOLO
import cv2

model = YOLO("yolov10s.pt")
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, imgsz=960, conf=0.5, iou = 0.5)
    cv2.imshow("YOLO", results[0].plot())

    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
        break

    if cv2.getWindowProperty("YOLO", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
