import cv2
import numpy as np

img1 = cv2.imread('img.jpg')
img2 = cv2.imread('img.jpg')


h, w, _ = img1.shape

for h1 in range(h-1, -1, -1):
    # 1. Lay phan tren
    img1_sub = img1[h1:]
    # 2. Lay phan duoi
    img2_sub = img2[0:h1]
    # 3. Gop 2 anh
    img_final = np.concatenate((img1_sub, img2_sub), axis=0)
    
    cv2.imshow('img', img_final)
    cv2.waitKey(1)



