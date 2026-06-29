from ultralytics import YOLO
import cv2

# Load mô hình YOLOv8 đã được huấn luyện sẵn
model = YOLO("yolov8n.pt")

# Đường dẫn ảnh cần nhận diện
image_path = "dataset/test.jpg"

# Thực hiện Object Detection
results = model.predict(
    source=image_path,
    conf=0.5,
    save=True,
    show=True
)

# Hiển thị thông tin các đối tượng phát hiện được
for result in results:

    print("Number of objects:", len(result.boxes))

    for box in result.boxes:

        cls = int(box.cls[0])

        confidence = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0]

        print("-------------------------")
        print("Class ID:", cls)
        print("Class Name:", model.names[cls])
        print("Confidence:", round(confidence, 3))
        print("Bounding Box:",
              int(x1),
              int(y1),
              int(x2),
              int(y2))

print("\nObject Detection Finished!")