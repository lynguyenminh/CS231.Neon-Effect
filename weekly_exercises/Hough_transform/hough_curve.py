import cv2
import numpy as np


def deg2grad(deg):
    return deg*3.141592654/180


# read image
img = cv2.imread('test.png', 0)
edges = cv2.Canny(img, 50, 150, apertureSize=3)
lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=100)

# Step 0.5: initialize hough table
theta_range = np.arange(-3.14, 3.14, 0.01)
H = np.zeros((500, len(theta_range)), dtype=np.uint8)

# Step 1: accumulate hough space
def accumulate(point):
    #for theta in range(360):
    for theta in theta_range:
        pro = point[0]*np.cos(theta) + point[1]*np.sin(theta)
        # If pro in range of Hough space
        if pro >= 0 and pro < 500:
            # map theta to Hough space
            H[int(pro), int((theta+3.14)/0.01)] += 1


h, w = edges.shape[:2]
for i in range(h):
    for j in range(w):
        if edges[i, j] != 0:
            accumulate([i, j])
lineK = np.where(H > 50)      



for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img,(x1,y1),(x2,y2),(0,0,0),1)    
cv2.imshow('a',img)


for i in range(len(lineK[0])):
    pro = lineK[0][i]
    theta = lineK[1][i] * 0.01 - 3.14
    a = np.cos(theta)
    b = np.sin(theta)
    if a ==0:
        x1, x2 = 0, 1000
        y1, y2 = int(pro / b), y1
    elif b == 0:
        x1, x2 = int(pro / a), x1
        y1, y2 = 0, 1000
    else: 
        x1, x2 = 0, 1000
        y1, y2 = int(pro/np.sin(theta)),int((pro - x2 * a)/b)
    cv2.line(img,(y1,x1),(y2,x2),(0,0,255),1)    
cv2.imshow('result', img)
cv2.waitKey(0)