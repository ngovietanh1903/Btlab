import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# ĐỌC ẢNH
# ==========================
img = cv.imread("messi5.jpg", cv.IMREAD_GRAYSCALE)

# ==========================
# TỰ VIẾT SOBEL X
# ==========================
sobel_kernel_x = np.array([
    [-1,0,1],
    [-2,0,2],
    [-1,0,1]
])

# Padding
pad = 1
padded = np.pad(img, pad, mode='edge')

custom_sobel = np.zeros_like(img)

rows, cols = img.shape

for i in range(rows):
    for j in range(cols):

        region = padded[i:i+3, j:j+3]

        value = np.sum(region * sobel_kernel_x)

        custom_sobel[i,j] = np.clip(abs(value),0,255)

# ==========================
# OPENCV
# ==========================

laplacian = cv.Laplacian(img, cv.CV_64F)

sobelx = cv.Sobel(img,
                  cv.CV_64F,
                  1, 0,
                  ksize=3)

sobely = cv.Sobel(img,
                  cv.CV_64F,
                  0, 1,
                  ksize=3)

# ==========================
# HIỂN THỊ
# ==========================

plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plt.imshow(custom_sobel,cmap='gray')
plt.title("Custom Sobel X")

plt.subplot(2,2,2)
plt.imshow(laplacian,cmap='gray')
plt.title("Laplacian")

plt.subplot(2,2,3)
plt.imshow(sobelx,cmap='gray')
plt.title("OpenCV Sobel X")

plt.subplot(2,2,4)
plt.imshow(sobely,cmap='gray')
plt.title("OpenCV Sobel Y")

plt.tight_layout()
plt.show()