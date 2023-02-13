import cv2
import numpy as np

# Read image
image = cv2.imread('im.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert binary image
_, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
img = 255 - bw


#  dung dung erosion de tach cac te bao
kernel = np.ones((10, 1),np.uint8)
img = cv2.erode(img, kernel, iterations = 1)
kernel = np.ones((10, 1),np.uint8)
img = cv2.erode(img, kernel, iterations = 1)
kernel = np.ones((1, 5),np.uint8)
img = cv2.erode(img, kernel, iterations = 1)
kernel = np.ones((1, 5),np.uint8)
img = cv2.erode(img, kernel, iterations = 1)
kernel = np.ones((1, 5),np.uint8)
img = cv2.erode(img, kernel, iterations = 1)
kernel = np.ones((1, 5),np.uint8)
img = cv2.erode(img, kernel, iterations = 1)
kernel = np.ones((1, 5),np.uint8)
img = cv2.erode(img, kernel, iterations = 1)
kernel = np.ones((3, 1),np.uint8)
img = cv2.erode(img, kernel, iterations = 1)
kernel = np.ones((3, 1),np.uint8)
img = cv2.erode(img, kernel, iterations = 1)
kernel = np.ones((5, 1),np.uint8)
img = cv2.erode(img, kernel, iterations = 1)


cv2.imshow('test', img)

# Find Canny edges
edged = cv2.Canny(img, 30, 200)

# Finding Contours
contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Draw all contours
# -1 signifies drawing all contours
cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

cv2.imshow('Contours', image)
cv2.waitKey(0)

print("Number of Contours found = " + str(len(contours)))
