import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh grayscale
query = cv.imread('messi5.jpg', 0)

img1 = cv.imread('img1.jpg', 0)
img2 = cv.imread('img2.jpg', 0)
img3 = cv.imread('img3.jpg', 0)

images = [img1, img2, img3]
names = ['img1', 'img2', 'img3']

# =========================
# RAW DATA RETRIEVAL
# =========================
print("=== RAW DATA COMPARISON ===")

for i in range(len(images)):

    # Resize cho cùng kích thước
    img = cv.resize(images[i], (query.shape[1], query.shape[0]))

    # Tính sai khác pixel
    diff = np.sum(np.abs(query.astype("float") - img.astype("float")))

    print(names[i], "Difference =", diff)

# =========================
# HISTOGRAM RETRIEVAL
# =========================
print("\n=== HISTOGRAM COMPARISON ===")

# Histogram ảnh gốc
hist_query = cv.calcHist([query], [0], None, [256], [0,256])

for i in range(len(images)):

    img = cv.resize(images[i], (query.shape[1], query.shape[0]))

    # Histogram ảnh phụ
    hist_img = cv.calcHist([img], [0], None, [256], [0,256])

    # So sánh histogram
    score = cv.compareHist(hist_query, hist_img, cv.HISTCMP_CORREL)

    print(names[i], "Similarity =", score)

# =========================
# HIỂN THỊ ẢNH
# =========================
cv.imshow("Query Image", query)
cv.imshow("Image 1", img1)
cv.imshow("Image 2", img2)
cv.imshow("Image 3", img3)

cv.waitKey(0)
cv.destroyAllWindows()