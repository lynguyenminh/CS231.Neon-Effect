import cv2
import numpy as np

img1 = cv2.imread('img.jpg')

img2 = cv2.imread('img.jpg')

h, w, _ = img1.shape


for D in range(0, w, 1):
    # 1. Lay anh trai
    sub_img1 = img1[:, D:]
    # 2. Lay anh phai
    sub_img2 = img2[:, :D]
    # gop anh
    img_final = np.concatenate((sub_img1, sub_img2), axis=1)

    cv2.imshow('img', img_final)
    cv2.waitKey(1)