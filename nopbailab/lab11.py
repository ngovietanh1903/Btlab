import cv2 as cv
import numpy as np

# Đọc ảnh xám
img = cv.imread("messi5.jpg", 0)

if img is None:
    print("Không đọc được ảnh")
    exit()

# Tạo kernel 5x5
kernel = np.ones((5,5), np.uint8)

# 1. Erosion
erosion = cv.erode(img, kernel, iterations=1)

# 2. Dilation
dilation = cv.dilate(img, kernel, iterations=1)

# 3. Opening
opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

# 4. Closing
closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

# 5. Gradient
gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)

# 6. Top Hat
tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)

# 7. Black Hat
blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)

# Hiển thị kết quả
cv.imshow("Original", img)
cv.imshow("Erosion", erosion)
cv.imshow("Dilation", dilation)
cv.imshow("Opening", opening)
cv.imshow("Closing", closing)
cv.imshow("Gradient", gradient)
cv.imshow("Top Hat", tophat)
cv.imshow("Black Hat", blackhat)

cv.waitKey(0)
cv.destroyAllWindows()