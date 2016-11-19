# -*- coding: UTF-8 -*-
import time
import cv2

URLTEST     = 'Source/Bernardo/TestDatabase/'
URLTRAIN    = 'Source/Bernardo/TrainDatabase/'
URLOTHERS   = 'Source/OthersImages/'
EXTENSION   = '.jpg'

def grayScale(image):
    try:
        M, N, O = image.shape

        grayImage = np.zeros((M, N, 1), dtype=np.uint8)
        initialTime = time.time()
        for i in range(M):
            for j in range(N):
                grayImage[i][j][0] = image[i][j][0] * 0.2989 + image[i][j][1] * 0.5870 + image[i][j][2] * 0.1140

        return grayImage

    except:
        return image


if __name__ == '__main__':
    from HistogramEqualization import *
    from LaplacianFilter import *
    from FourierTransform import *
    from FourierTransform2 import *
    from BruteForce import *
    from CompleteBruteForce import *
    from Person import *


    initialTime = time.time()

    image1 = cv2.imread(URLOTHERS+'game1.png')
    image2 = cv2.imread(URLOTHERS+'game2.png')
    #image = grayScale(image)

    #Here is another way to convert to grayscale using opencv
    #gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    # histogramEqualization = HistogramEqualization(image)
    # equalizedImage = histogramEqualization.calculate(True)

    # laplacian = LaplacianFilter(image)
    # laplacian.laplacianFilter(True)

    # fourier = FourierTransform(image)
    # fourier.fourierTransform(1, 50, True)

    completeBrute = CompleteBruteForce(urlTestImage=URLTEST+'10'+EXTENSION)
    foundPerson, percentage = completeBrute.bruteForce()

    print 'The person found was:', foundPerson.getName(), 'with ', percentage*100, '% of accuracy'

    print 'Past time:', time.time() - initialTime
