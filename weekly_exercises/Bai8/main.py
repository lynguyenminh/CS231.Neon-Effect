import cv2
import numpy as np

# 1. load anh grayscale
img = cv2.imread('im1.jpg', 0)
img2 = img.copy()

# 2. tinh histogram
(pixel, hist_matrix) = np.unique(img, return_counts=True)

# 3. tinh tong tich luy CDF
cdf = [0]
for i, _ in enumerate(pixel): 
    cdf.append(cdf[-1] + hist_matrix[i])
cdf = cdf[1:]

min_value = cdf[0]
max_value = cdf[-1]
cdf = [(i - min_value)/(max_value - min_value) for i in cdf]
cdf = np.array(cdf)
cdf *= 255
cdf = cdf.astype(int)

# 4. Mapping image
for i, value in enumerate(pixel): 
    img2[img == value] = cdf[i]
    
# 5. show anh
cv2.imshow('', img2)
cv2.waitKey()