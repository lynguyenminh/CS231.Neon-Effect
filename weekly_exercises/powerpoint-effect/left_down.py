import cv2
import numpy as np

img1 = cv2.imread('img.jpg')
img2 = cv2.imread('img.jpg')


h, w, _ = img1.shape
print(w, h)

for percent in range(99, 0, -1):
    sub_w, sub_h = int(w*percent/100), int(h * percent/100)

    # 1. Lay anh nho
    sub_img1 = img1[sub_h:, sub_w:]
    # 2. Ghi de len anh lon
    img2[:h - sub_h, sub_w:] = sub_img1
    # 3. Show anh
    cv2.imshow('img', img2)
    cv2.waitKey(1)