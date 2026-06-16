# LAB 10 - THRESHOLDING (PHÂN NGƯỠNG ẢNH)
# Thresholding là kỹ thuật phân ngưỡng dùng để chuyển ảnh xám thành ảnh nhị phân (đen trắng).
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

import cv2 as cv
import numpy as np

# Đọc ảnh ở chế độ ảnh xám (grayscale)
img = cv.imread("messi5.jpg", 0)

# Kiểm tra đọc ảnh
if img is None:
    print("Không đọc được ảnh!")
    exit()


# PHẦN 1: TỰ VIẾT THRESHOLDING

#
# Ý tưởng:
# - Nếu pixel > 127 -> gán 255 (trắng)
# - Nếu pixel <= 127 -> gán 0 (đen)
#
# Công thức:
#
# f(x,y) > T  => 255
# f(x,y) <= T => 0
#


threshold = 127

rows, cols = img.shape

my_threshold = np.zeros((rows, cols), dtype=np.uint8)

for i in range(rows):
    for j in range(cols):

        if img[i, j] > threshold:
            my_threshold[i, j] = 255
        else:
            my_threshold[i, j] = 0


# PHẦN 2: THRESHOLDING BẰNG OPENCV

#
# cv.threshold(
#     ảnh nguồn,
#     giá trị ngưỡng,
#     giá trị max,
#     kiểu threshold
# )
#
# ==================================================

ret, opencv_threshold = cv.threshold(
    img,
    127,
    255,
    cv.THRESH_BINARY
)


# PHẦN 3: ADAPTIVE THRESHOLDING

#
# Dùng khi ảnh có vùng sáng tối khác nhau.
# Mỗi vùng sẽ có ngưỡng riêng.
#
# blockSize = 11
# C = 2


adaptive = cv.adaptiveThreshold(
    img,
    255,
    cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv.THRESH_BINARY,
    11,
    2
)


# PHẦN 4: OTSU THRESHOLDING

#
# Otsu tự động tìm giá trị ngưỡng tối ưu
# từ Histogram của ảnh.
#
# Không cần tự chọn threshold.
#


ret_otsu, otsu = cv.threshold(
    img,
    0,
    255,
    cv.THRESH_BINARY + cv.THRESH_OTSU
)

print("Ngưỡng Otsu tìm được =", ret_otsu)


# HIỂN THỊ KẾT QUẢ


cv.imshow("Original Image", img)

cv.imshow("1. My Threshold", my_threshold)

cv.imshow("2. OpenCV Threshold", opencv_threshold)

cv.imshow("3. Adaptive Threshold", adaptive)

cv.imshow("4. Otsu Threshold", otsu)

cv.waitKey(0)
cv.destroyAllWindows()