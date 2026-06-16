import cv2 as cv
import numpy as np

# Mở webcam
cap = cv.VideoCapture(0)

while True:

    # Đọc từng frame
    ret, frame = cap.read()

    if not ret:
        print("Không mở được camera")
        break

    # Chuyển BGR sang HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Khoảng màu xanh dương trong HSV
    lower_blue = np.array([0,120,70])
    upper_blue = np.array([10,255,255])

    # Tạo mask màu xanh
    mask = cv.inRange(hsv, lower_blue, upper_blue)

    # Lấy riêng vật màu xanh
    res = cv.bitwise_and(frame, frame, mask=mask)

    # Hiển thị
    cv.imshow('Frame', frame)
    cv.imshow('Mask', mask)
    cv.imshow('Result', res)

    # Nhấn ESC để thoát
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

# Giải phóng camera
cap.release()
cv.destroyAllWindows()