import cv2

source = cv2.imread('source.jpeg')
dst = cv2.imread('dest.jpeg')


source = cv2.resize(source, (250, 250))
dst = cv2.resize(dst, (250, 250))


result = dst.copy()
for f in range(3 * 24):
    t = f/(3 * 24.)
    result = cv2.addWeighted(dst, t, source, 1 - t, 0)

    cv2.imshow('', result)
    cv2.waitKey(100)
