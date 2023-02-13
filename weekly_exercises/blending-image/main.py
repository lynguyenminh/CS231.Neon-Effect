import numpy as np
import cv2


img1 = cv2.imread('im2.jpg')
img2 = cv2.imread('im2.png')

img1 = cv2.resize(img1, (100, 100))
img2 = cv2.resize(img2, (100, 100))

def alphaBlending(img1, img2, alpha=0.2):
    blending = img1 * alpha + img2 * (1- alpha)
    blending = blending.astype(np.uint8)
    return blending

blending = alphaBlending(img1, img2, alpha=0.5)


cv2.imshow('blending', blending)
cv2.waitKey(0)




