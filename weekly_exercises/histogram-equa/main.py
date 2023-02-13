import cv2
import numpy as np


img = cv2.imread('im1.jpg', 0)

# b1: Tinh histogram
def histogram(img):
    flat_img = img.ravel() #flatten image
    hist = np.zeros(256, dtype=np.int)
    np.add.at(hist, flat_img, 1) #add 1 in the histogram
    return hist

# b2: Tinh tong tich luy
def tong_tich_luy(img):
    hist = histogram(img)
    ma_tran_tich_luy = [hist[0]]
    for i in hist[1:]:
        ma_tran_tich_luy.append(i + ma_tran_tich_luy[-1])
    return ma_tran_tich_luy

# b3: can bang 
def equalization_histogram(img): 
    ma_tran_tich_luy = tong_tich_luy(img)
    ma_tran_tich_luy = [(i - ma_tran_tich_luy[0])/(ma_tran_tich_luy[-1] - ma_tran_tich_luy[0]) * 255 for i in ma_tran_tich_luy]
    ma_tran_tich_luy = list(map(int, ma_tran_tich_luy))
    return ma_tran_tich_luy

# B4: ap vao anh
def apply_CDF_function(img): 
    ma_tran_tich_luy = equalization_histogram(img)
    img_equalized = np.interp(img, np.arange(0, 256), ma_tran_tich_luy).astype(np.uint8)
    return img_equalized

result = apply_CDF_function(img)
cv2.imshow('result', result)
cv2.waitKey(0)

# mở rộng cho ảnh màu, có 2 cách để thực hiện: 
# 1. Thay đổi trên V channel của HSV (Nhanh hơn)
# 2. Thay đổi trên 3 channel của RGB
# C1:
# img = cv2.imread('im3.png')
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# img[..., 2] = apply_CDF_function(img[..., 2])
# cv2.imshow('img', img)
# cv2.waitKey(0)

# C2: 
# img = cv2.imread('im3.png')

# for i in range(2):
#     img[..., i] = apply_CDF_function(img[..., i])
# cv2.imshow('img', img)
# cv2.waitKey(0)
