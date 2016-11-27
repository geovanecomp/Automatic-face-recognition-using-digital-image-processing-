import cv2

def readImage(urlImage, channels=3):
    image = cv2.imread(urlImage, channels)
    if channels == 0:
        M, N = image.shape
        O = 1
        return image.reshape((M,N,O))
    return image
