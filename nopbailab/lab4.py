import cv2 as cv
import numpy as np

# Đọc ảnh grayscale ( ảnh xám)
# Ảnh xám là ảnh chỉ còn mức sáng tối từ đen đến trắng, không còn màu
img = cv.imread('messi5.jpg', 0)

# Kiểm tra ảnh
if img is None:
    print("Không đọc được ảnh")
    exit()

# 1.Tạo ảnh Âm bản
negative = 255 - img
#  Ảnh grayscale 8-bit có giá trị lớn nhất là 255.

# 2. Tăng sáng ảnh
bright = cv.add(img, 50)

# 3. Giảm sáng ảnh
dark = cv.subtract(img, 50)

# 4. Threshold nhị phân
_, thresh = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

# Hiển thị kết quả
cv.imshow('Original', img)
cv.imshow('Negative', negative)
cv.imshow('Bright', bright)
cv.imshow('Dark', dark)
cv.imshow('Threshold', thresh)

cv.waitKey(0)
cv.destroyAllWindows()