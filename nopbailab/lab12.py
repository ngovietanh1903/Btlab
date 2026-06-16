# Contour (đường bao) là đường nối các điểm liên tiếp có cùng cường độ sáng,
# Nó được sử dụng để xác định hình dạng, kích thước và vị trí của vật thể.

# Mục đích:
# Tách vật thể khỏi nền.
# Đo kích thước.
# Nhận dạng hình dạng.

# Contours được dùng trong:
# Nhận dạng vật thể
# Đếm xe
# Đếm sản phẩm
# Nhận dạng biển số
# Nhận dạng chữ viết
# Theo dõi chuyển động
# 
# Contour là kỹ thuật tìm đường biên của đối tượng trong ảnh.
# Trong OpenCV, contour được tìm bằng hàm cv.findContours(), sau đó có thể tính diện tích,
# chu vi, hình chữ nhật bao, convex hull và các đặc trưng khác để phục vụ nhận dạng và xử lý ảnh.
# 
# Lap 12 thực hiện nhị phân hóa ảnh, tìm contour bằng cv.findContours(), vẽ contour bằng cv.drawContours() 
# và tính các đặc trưng như diện tích, chu vi và hình chữ nhật bao quanh đối tượng.

import cv2 as cv
import numpy as np

# Đọc ảnh
img = cv.imread("messi5.jpg")

# Kiểm tra ảnh
if img is None:
    print("Khong doc duoc anh!")
    exit()

# Chuyển sang ảnh xám
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Tách ngưỡng (Threshold)
ret, thresh = cv.threshold(
    gray,
    127,
    255,
    cv.THRESH_BINARY
)

# Loại bỏ bớt nhiễu
kernel = np.ones((3, 3), np.uint8)

thresh = cv.morphologyEx(
    thresh,
    cv.MORPH_OPEN,
    kernel
)

# Tìm contour
contours, hierarchy = cv.findContours(
    thresh,
    cv.RETR_TREE,
    cv.CHAIN_APPROX_SIMPLE
)

print("So contour tim duoc:", len(contours))

# Lấy contour có diện tích lớn nhất
cnt = max(contours, key=cv.contourArea)

# Tính diện tích
area = cv.contourArea(cnt)
print("Dien tich =", area)

# Tính chu vi
perimeter = cv.arcLength(cnt, True)
print("Chu vi =", perimeter)

# Tạo ảnh để vẽ contour
draw = img.copy()

# Vẽ contour lớn nhất màu xanh lá
cv.drawContours(
    draw,
    [cnt],
    -1,
    (0, 255, 0),
    2
)

# Hình chữ nhật bao quanh contour
x, y, w, h = cv.boundingRect(cnt)

cv.rectangle(
    draw,
    (x, y),
    (x + w, y + h),
    (0, 0, 255),
    2
)

# Hiển thị ảnh
cv.imshow("Original", img)
cv.imshow("Threshold", thresh)
cv.imshow("Largest Contour", draw)

cv.waitKey(0)
cv.destroyAllWindows()