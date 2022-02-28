import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt

# 伽玛变换  power函数实现幂函数
def hist_equal(img, z_max=255):

    H, W = img.shape
    S = H * W * 1.

    out = img.copy()

    sum_h = 0.

    for i in range(1, 255):
        ind = np.where(img == i)
        sum_h += len(img[ind])
        z_prime = z_max / S * sum_h
        out[ind] = z_prime

    out = out.astype(np.uint8)

    return out

if __name__ == "__main__":
    imageData = cv2.imread("E:\workspace\pycharm\pyqt5-imageprocessing\imgs\\thorino.jpg")
    # 归1
    # imageData = cv2.cvtColor(imageData, cv2.COLOR_BGR2GRAY)
    # imageData = cv2.cvtColor(imageData, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(imageData, cv2.COLOR_BGR2GRAY)

    thresh, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    result = np.stack((out,) * 3, axis=-1)

    print(f'result.shape {result.shape}')
    print(f'result {result}')
    #效果
    cv2.imshow('img',imageData)
    cv2.imshow('O',result)
    # cv2.imwrite("O.png", O)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite(filePath, outputImageData)

    # result = np.stack((result,) * 3, axis=-1)

