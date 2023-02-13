# phase 1 = Tao du lieu =================================
import numpy as np
import cv2

background_dataset = cv2.imread('background.png')
foreground_dataset = cv2.imread('foreground.jpg')

background_dataset = np.array([background_dataset[..., i].ravel() for i in range(3)])
foreground_dataset = np.array([foreground_dataset[..., i].ravel() for i in range(3)])

x_train = np.concatenate((background_dataset, foreground_dataset), axis = 1).T
y_train = np.concatenate((np.ones((background_dataset.shape[1], 1)), np.zeros((foreground_dataset.shape[1], 1))), axis = 0)

print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")

# phase 2 = Train model =================================
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0).fit(x_train, y_train)

# Phase 3 = apply =================================
# tao mask
test_img = cv2.imread('test_img.jpg')
origin_shape = test_img.shape
temp = test_img.copy()

test_img = np.array([test_img[..., i].ravel() for i in range(3)]).T
predict = clf.predict(test_img)

mask = predict.reshape(origin_shape[0], origin_shape[1]).astype(np.uint8)
mask = 1 - mask


# lam như bth
result = cv2.bitwise_and(temp, temp, mask=mask)


cv2.imshow('results', result)
cv2.waitKey(0)
