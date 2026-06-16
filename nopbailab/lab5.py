# Histogram là biểu đồ phân bố mức sáng của ảnh.
# Dùng để phân tích độ sáng, độ tương phản và phân bố cường độ sáng của ảnh.
# Trục X: mức xám từ 0 → 255
# Trục Y: số lượng pixel ở mức xám đó
# 
# Histogram được dùng để:
# Phân tích độ sáng ảnh
# Cải thiện độ tương phản
# Thresholding
# Nhận dạng ảnh
# Tiền xử lý trong Computer Vision
# 
# 
# Nhược điểm của Histogram
# 1. Không chứa thông tin vị trí pixel

# Histogram chỉ cho biết có bao nhiêu pixel ở mỗi mức xám.
# Không cho biết các pixel đó nằm ở đâu trong ảnh.
# 

# 2. Không mô tả được hình dạng vật thể

# Histogram chỉ phản ánh độ sáng.
# Không cho biết ảnh có hình tròn, hình vuông hay khuôn mặt người.
# 
# 3. Nhạy với điều kiện chiếu sáng

# Khi ánh sáng thay đổi, histogram cũng thay đổi đáng kể.
# Cùng một vật thể nhưng chụp sáng hơn hoặc tối hơn sẽ cho histogram khác.



# Lab 5 thực hiện tính Histogram của ảnh xám. Chương trình đọc ảnh, tạo mảng 256 phần tử tương
# ứng 256 mức xám, duyệt từng pixel để đếm số lần xuất hiện và dùng Matplotlib để vẽ biểu đồ Histogram. 


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# Đọc ảnh grayscale
img = cv.imread('messi5.jpg', 0)

# Kiểm tra ảnh
if img is None:
    print("Không đọc được ảnh")
    exit()

# Tạo mảng histogram 256 mức xám
hist = [0] * 256

# Duyệt từng pixel và đếm
for row in img:
    for pixel in row:
        hist[pixel] += 1

# Hiển thị ảnh
cv.imshow('Original Image', img)

# Vẽ histogram
plt.plot(hist)
plt.title('Histogram')
plt.xlabel('Gray Level')
plt.ylabel('Number of Pixels')

plt.show()

cv.waitKey(0)
cv.destroyAllWindows()