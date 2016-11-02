# -*- coding: UTF-8 -*-
import time
import cv2
if __name__ == '__main__':
    from HistogramEqualization import *
    from LaplacianFilter import *
    from FourierTransform import *

    image = cv2.imread('Source/OthersImages/Cameraman.tif', 0)
    initialTime = time.time()
    #
    # histogramEqualization = HistogramEqualization(image)
    # equalizedImage = histogramEqualization.calculate(True)
    #
    #
    # laplacian = LaplacianFilter(equalizedImage)
    # laplacian.laplacianFilter(True)

    fourier = FourierTransform(image)
    fourier.FourierTransform(1, 50, True)


    print 'Past time:', time.time() - initialTime
