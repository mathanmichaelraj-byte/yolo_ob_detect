from ultralytics import YOLO
import cv2

model = YOLO('yolov10s.pt')
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret ,frame = cap.read()

    if not ret:
        break

    results = model(frame)

    cv2.imshow("YOLO", results[0].plot())

    if cv2.waitKey(27) & 0xFF == ord('q'):
        break

    if cv2.getWindowProperty("YOLO", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release() 
cv2.destroyAllWindows()