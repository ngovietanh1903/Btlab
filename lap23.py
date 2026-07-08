from ultralytics import YOLO
import cv2

# Load mô hình YOLOv8
model = YOLO("yolov8n.pt")

# Đọc video
cap = cv2.VideoCapture("video.mp4")

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    # Theo dõi đối tượng
    results = model.track(
        frame,
        persist=True,
        conf=0.5
    )

    # Lấy thông tin các đối tượng
    boxes = results[0].boxes

    if boxes.id is not None:

        ids = boxes.id.cpu().numpy()
        classes = boxes.cls.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        coords = boxes.xyxy.cpu().numpy()

        for i in range(len(ids)):

            x1, y1, x2, y2 = coords[i]

            print("---------------------------")
            print("Tracking ID :", int(ids[i]))
            print("Class       :", model.names[int(classes[i])])
            print("Confidence  :", round(float(confs[i]), 3))
            print("BoundingBox :", int(x1), int(y1), int(x2), int(y2))

    # Vẽ khung và ID lên video
    annotated_frame = results[0].plot()

    # Hiển thị video
    cv2.imshow("Object Tracking", annotated_frame)

    # Nhấn phím Q để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng bộ nhớ
cap.release()
cv2.destroyAllWindows()

print("Object Tracking Finished!")