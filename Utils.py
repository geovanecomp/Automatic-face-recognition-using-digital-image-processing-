import cv2
import numpy as np

def readImage(urlImage, channels=3):
    image = cv2.imread(urlImage, channels)
    if channels == 0:
        M, N = image.shape
        O = 1
        return image.reshape((M,N,O))
    return image

#Transform all values bigger than max value (and min) to be equal max value (and min)
def correctMatrixValues(matrix, maxValue=255, minValue=0):
    try:
        M,N,O = matrix.shape
    except:
        M,N = matrix.shape
        O = 1
        matrix = matrix.reshape((M,N,O))

    for i in range(M):
        for j in range(N):
            for k in range(O):
                if matrix[i][j][k] > maxValue:
                    matrix[i][j][k] = maxValue

                if matrix[i][j][k] < minValue:
                    matrix[i][j][k] = minValue

    if maxValue == 255:
        matrix = np.uint8(matrix)
    elif maxValue == 65535:
        matrix = np.uint16(matrix)

    return matrix


def showImage(image):

    image = correctMatrixValues(image)

    cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', 1200, 600)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
