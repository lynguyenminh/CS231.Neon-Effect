import numpy as np
import cv2


img = cv2.imread('test_img.jpg')
background = cv2.imread('background.png')

min_value = np.array([background[..., i].min() for i in range(3)]).astype(int) - 100
max_value = np.array([background[..., i].max() for i in range(3)]).astype(int) + 100


mask = cv2.inRange(img, min_value, max_value)

img = cv2.bitwise_and(img, img, mask=~mask)

cv2.imshow('results', img)
cv2.waitKey(0)

print(mask.max(), mask.min())
print(mask.dtype)
