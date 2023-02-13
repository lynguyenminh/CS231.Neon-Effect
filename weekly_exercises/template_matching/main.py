import cv2
import numpy as np


def correlation(img, kernel):
    xOutput=int((img.shape[0]-kernel.shape[0])+1)
    yOutput=int((img.shape[1]-kernel.shape[1])+1)

    output=np.zeros((xOutput,yOutput))

    for i in range(xOutput):
        for j in range(yOutput):
            output[i,j] = np.sum(np.multiply(kernel,img[i:i+kernel.shape[0],j:j+kernel.shape[1]]))
            
    return output


if __name__=="__main__":
    # 1. Read images
    img = cv2.imread('9-ro.jpeg', 0)
    kernel = cv2.imread('template.png', 0)


    # normalize image
    img = img / 255.
    kernel = kernel / 255.


    cor = correlation(img, kernel)
    cor = cor.astype(np.uint8)

    img = img[kernel.shape[0]//2: kernel.shape[0]//2 + cor.shape[0], kernel.shape[1]//2: kernel.shape[1]//2 + cor.shape[1]]

    th, thresh = cv2.threshold(cor, 127, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    

    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        rect = cv2.boundingRect(c)
        if rect[2] < 70 or rect[2] > 100 or rect[3] > 100 or rect[3] < 70: continue
        x,y,w,h = rect
        boxes.append((x, y, w, h))

    boxes = np.array(boxes)
    scores = np.array([0.8] * len(boxes))

    idxs = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.4, nms_threshold=0.1)
    for id in idxs:
        x, y, w, h = boxes[id]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, "ID: " + str(id), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)