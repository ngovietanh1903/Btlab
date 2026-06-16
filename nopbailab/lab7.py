# Tích chập là phép nhân và cộng giữa kernel với vùng ảnh để tạo ảnh mới.
# Mục đích của tích chập:
# Làm mờ ảnh
# Khử nhiễu
# Làm sắc nét
# Phát hiện cạnh
# Trích xuất đặc trưng
# Kernel là Ma trận nhỏ dùng để lọc ảnh.


import cv2 as cv
import numpy as np

# Đọc ảnh ở chế độ grayscale
img = cv.imread("messi5.jpg", 0)

# Kiểm tra đọc ảnh thành công hay không
if img is None:
    print("Không đọc được ảnh!")
    exit()

# Lấy kích thước ảnh
rows, cols = img.shape

# Tạo kernel tích chập
# Tổng các phần tử = 1
# kernel chuẩn hóa
kernel = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
]) / 9

# Tạo ảnh kết quả cùng kích thước với ảnh gốc
output = np.zeros((rows, cols), dtype=np.uint8)


# THỰC HIỆN PHÉP TÍCH CHẬP

for i in range(1, rows - 1):
    for j in range(1, cols - 1):

        # Lấy vùng ảnh 3x3 xung quanh pixel hiện tại
        region = img[i - 1:i + 2, j - 1:j + 2]

        # Nhân từng phần tử với kernel rồi cộng lại
        value = np.sum(region * kernel)

        # Đảm bảo giá trị nằm trong khoảng 0-255
        output[i, j] = np.clip(value, 0, 255)

# ==========================================
# HIỂN THỊ KẾT QUẢ
# ==========================================
cv.imshow("Original Image", img)
cv.imshow("Convolution Result", output)

cv.waitKey(0)
cv.destroyAllWindows()

img = cv.imread("messi5.jpg", 0)

print(img is None)