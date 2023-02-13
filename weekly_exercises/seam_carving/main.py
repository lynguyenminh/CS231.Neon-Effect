import numpy as np
from numpy import sqrt
import cv2
import matplotlib.pyplot as plt


def cal_energy(arr):
    energy_x = np.concatenate((arr[:, 1:], np.zeros((arr.shape[0], 1))), axis=1) - arr
    energy_y = np.concatenate((arr[1:, :], np.zeros((1, arr.shape[1]))), axis=0) - arr
    energy = sqrt(energy_x **2 + energy_y **2)
    return energy


def seam_carving(arr):
    energy = cal_energy(arr)
    for i in range(1, energy.shape[0]): 
        for j in range(0, energy.shape[1]):
            if j == 0: 
                energy[i, j] += min(energy[i -1, :2])
            elif j == energy.shape[1]:
                energy[i, j] += min(energy[i -1, -2:]) 
            else: 
                energy[i, j] += min(energy[i -1, j-1:j+2])

    # khoi tao
    minimum_seam_index = []
    minimum_seam_index.append(np.argmin(energy[-1, :]))

    # 2. Tim vi tri min cac hang phia tren (vi tri i-1:i+1)
    for i in range(energy.shape[0] - 2, -1, -1):
        if minimum_seam_index[-1] == 0:
            minimum_seam_index.append(minimum_seam_index[-1] + np.argmin(energy[i, :2]))
        elif minimum_seam_index[-1] == arr.shape[1] - 1:
            minimum_seam_index.append(minimum_seam_index[-1] + np.argmin(energy[i, -2:]) -1)
        else:
            minimum_seam_index.append(minimum_seam_index[-1] + np.argmin(energy[i, minimum_seam_index[-1] -1: minimum_seam_index[-1] + 2]) -1)

    # 3. Dieu chinh lai vi tri minimum_seam_index phu hop voi anh goc
    minimum_seam_index.reverse()

    # 4. xoa cac gia tri tren duong seam tim duoc
    m,n = arr.shape
    arr = arr[np.arange(n) != np.array(minimum_seam_index)[:,None]].reshape(m,-1)

    return arr



if __name__=="__main__":    
    arr = cv2.imread('test.jpg', 0)
    for i in range(300):
        arr = seam_carving(arr)

    plt.imshow(arr)
    plt.show()