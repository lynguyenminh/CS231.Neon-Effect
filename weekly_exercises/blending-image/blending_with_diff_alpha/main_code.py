import imageio.v3 as iio
import imageio
import cv2
import numpy as np
import math

# 1. Read effect
frames = iio.imread("smoke02.gif", index=None)


# 2. Read foreground and mask
fg = cv2.imread('people.jpg')
fg = cv2.cvtColor(fg, cv2.COLOR_BGR2RGB)
fg = cv2.resize(fg, (500, 500))
fg = fg[50:-50, 50:-50, :]

mask = cv2.imread('people-mask.png', cv2.IMREAD_UNCHANGED)
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
mask = mask[50:-50, 50:-50, :]

# 3. Crop effect for compatable 
hg, wg, _ = fg.shape
_, he, we, _ = frames.shape

top = abs(hg - he)//2
left = abs(wg - we)//2

frames = frames[:, top:he - top, left:we-left, :]



# 4. Blending image
# 4.1. Chon ham sigmoid de chuan hoa alpha
def sigmoid(x):
    y = 1 / (1 + math.exp(9-18*x))
    y = round(y, 2)
    return y

# 4.2. Blending image
list_frame = []
for id, frame in enumerate(frames): 
    print("Handle frame: ", id)
    result = fg.copy()
    alpha_channel = mask[:, :, -1]
    for index1, i in enumerate(alpha_channel): 
        alpha_object = sigmoid(1- index1/alpha_channel.shape[0])
        alpha_background = 0
        for index2, j in enumerate(i): 
            if j != 0: 
                result[index1, index2]  = result[index1, index2] * alpha_object + (1- alpha_object) * frame[index1, index2]
            else: 
                result[index1, index2]  = frame[index1, index2]

    list_frame.append(result)

# 4. Save image
imageio.mimsave('result.gif', list_frame)