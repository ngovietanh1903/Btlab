import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("messi5.jpg")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# ==========================
# CUSTOM AVERAGE FILTER
# ==========================
def average_filter(image, ksize=5):

    pad = ksize // 2

    padded = np.pad(
        image,
        ((pad,pad),(pad,pad),(0,0)),
        mode='edge'
    )

    output = np.zeros_like(image)

    h,w,c = image.shape

    for y in range(h):
        for x in range(w):
            for ch in range(c):

                region = padded[
                    y:y+ksize,
                    x:x+ksize,
                    ch
                ]

                output[y,x,ch] = np.mean(region)

    return output

custom_avg = average_filter(img,5)

# ==========================
# OPENCV FILTERS
# ==========================
avg = cv.blur(img,(5,5))
gauss = cv.GaussianBlur(img,(5,5),0)
median = cv.medianBlur(img,5)
bilateral = cv.bilateralFilter(img,9,75,75)

# ==========================
# DISPLAY
# ==========================
plt.figure(figsize=(15,8))

plt.subplot(231)
plt.imshow(img)
plt.title("Original")

plt.subplot(232)
plt.imshow(custom_avg)
plt.title("Custom Average")

plt.subplot(233)
plt.imshow(avg)
plt.title("OpenCV Average")

plt.subplot(234)
plt.imshow(gauss)
plt.title("Gaussian")

plt.subplot(235)
plt.imshow(median)
plt.title("Median")

plt.subplot(236)
plt.imshow(bilateral)
plt.title("Bilateral")

plt.tight_layout()
plt.show()