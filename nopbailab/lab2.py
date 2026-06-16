import cv2 as cv
import numpy as np

# Đọc ảnh
img1 = cv.imread('messi5.jpg')
img2 = cv.imread('opencv-logo-white.png')

# Kiểm tra ảnh
if img1 is None or img2 is None:
    print("Không đọc được ảnh")
    exit()

# Resize logo nhỏ lại
img2 = cv.resize(img2, (150, 150))

# Lấy kích thước logo
rows, cols = img2.shape[:2]

# Tạo ROI ở góc trên trái
roi = img1[0:rows, 0:cols]

# Chuyển logo sang grayscale
img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# Tạo mask
ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)

# Đảo mask
mask_inv = cv.bitwise_not(mask)

# Làm nền vùng logo
img1_bg = cv.bitwise_and(roi, roi, mask=mask_inv)

# Lấy phần logo
img2_fg = cv.bitwise_and(img2, img2, mask=mask)

# Ghép lại
dst = cv.add(img1_bg, img2_fg)

# Gán vào ảnh chính
img1[0:rows, 0:cols] = dst

# Hiển thị
cv.imshow('Result', img1)

cv.waitKey(0)
cv.destroyAllWindows()