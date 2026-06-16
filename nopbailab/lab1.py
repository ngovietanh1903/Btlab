import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Đọc ảnh
img = cv.imread('messi5.jpg')

# Kiểm tra ảnh
if img is None:
    print("Không đọc được ảnh")
    exit()

# =========================
# Truy cập pixel
# =========================
px = img[100, 100]
print("Pixel:", px)

blue = img[100, 100, 0]
print("Blue:", blue)

# Sửa pixel
img[100, 100] = [255, 255, 255]

# =========================
# Thuộc tính ảnh
# =========================
print("Shape:", img.shape)
print("Size:", img.size)
print("Datatype:", img.dtype)

# =========================
# ROI
# =========================
ball = img[50:120, 50:120]
img[120:190, 150:220] = ball

# =========================
# Tách kênh
# =========================
b, g, r = cv.split(img)

# =========================
# Border
# =========================
BLUE = [255, 0, 0]

replicate = cv.copyMakeBorder(
    img,
    10,10,10,10,
    cv.BORDER_REPLICATE
)

reflect = cv.copyMakeBorder(
    img,
    10,10,10,10,
    cv.BORDER_REFLECT
)

reflect101 = cv.copyMakeBorder(
    img,
    10,10,10,10,
    cv.BORDER_REFLECT_101
)

wrap = cv.copyMakeBorder(
    img,
    10,10,10,10,
    cv.BORDER_WRAP
)

constant = cv.copyMakeBorder(
    img,
    10,10,10,10,
    cv.BORDER_CONSTANT,
    value=BLUE
)

# =========================
# Hiển thị
# =========================
plt.subplot(231)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title("ORIGINAL")

plt.subplot(232)
plt.imshow(cv.cvtColor(replicate, cv.COLOR_BGR2RGB))
plt.title("REPLICATE")

plt.subplot(233)
plt.imshow(cv.cvtColor(reflect, cv.COLOR_BGR2RGB))
plt.title("REFLECT")

plt.subplot(234)
plt.imshow(cv.cvtColor(reflect101, cv.COLOR_BGR2RGB))
plt.title("REFLECT_101")

plt.subplot(235)
plt.imshow(cv.cvtColor(wrap, cv.COLOR_BGR2RGB))
plt.title("WRAP")

plt.subplot(236)
plt.imshow(cv.cvtColor(constant, cv.COLOR_BGR2RGB))
plt.title("CONSTANT")

plt.show()