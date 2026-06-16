import cv2 as cv
import numpy as np

img = cv.imread("img5.jpg")

if img is None:
    print("Khong doc duoc anh!")
    exit()

width = 900
height = int(img.shape[0] * width / img.shape[1])

img = cv.resize(img, (width, height))

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

_, thresh = cv.threshold(
    gray,
    140,
    255,
    cv.THRESH_BINARY_INV
)

kernel = np.ones((3, 3), np.uint8)

thresh = cv.morphologyEx(
    thresh,
    cv.MORPH_OPEN,
    kernel
)

contours, _ = cv.findContours(
    thresh,
    cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_SIMPLE
)

print("So contour tim duoc:", len(contours))

result = img.copy()

for cnt in contours:

    area = cv.contourArea(cnt)

    if area < 5000:
        continue

    peri = cv.arcLength(cnt, True)

    approx = cv.approxPolyDP(
        cnt,
        0.02 * peri,
        True
    )

    rect = cv.minAreaRect(cnt)

    rw = rect[1][0]
    rh = rect[1][1]

    if rw == 0 or rh == 0:
        continue

    ratio = max(rw, rh) / min(rw, rh)

    shape = ""

    # Hình chữ nhật dài
    if ratio > 1.5:
        shape = "Hinh Chu Nhat"

    else:
        # Gần vuông hoặc tròn
        circularity = 4 * np.pi * area / (peri * peri)

        if circularity > 0.80:
            shape = "Hinh Tron"
        else:
            shape = "Hinh Vuong"

    print("Vat the:", shape)

    x, y, w, h = cv.boundingRect(cnt)

    cv.drawContours(
        result,
        [approx],
        -1,
        (0, 255, 0),
        3
    )

    cv.putText(
        result,
        shape,
        (x, y - 10),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

cv.imshow("Threshold", thresh)
cv.imshow("Result", result)

cv.waitKey(0)
cv.destroyAllWindows()