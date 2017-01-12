# -*- coding: UTF-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

#To not abbreviate big matrices
np.set_printoptions(threshold='nan')

class FourierTransform2(object):
    def __init__(self, image):
        self.originalImage = np.float32(image)
        self.M, self.N, self.O = self.originalImage.shape #Number of lines and columns
        self.P = 2 * self.M
        self.Q = 2 * self.N
        self.transformedImage = np.zeros((self.P, self.Q, self.O), dtype=np.float32)
        for i in range(self.M):
            for j in range(self.N):
                for k in range(self.O):
                    self.transformedImage[i][j][k] = self.originalImage[i][j][k]

#To move the information in the corners to the center in the frequency domain---
    def __centeringImage(self, image):

        for i in range(self.P):
            for j in range(self.Q):
                for k in range(self.O):
                    image[i][j][k] = image[i][j][k] * (-1)**(i+j)

        return image

#Distance calculation to create the filters-------------------------------------
    def __distanceCalculation(self):

        distance = np.zeros((self.P, self.Q), dtype=np.float32)
        for i in range(self.P):
            for j in range(self.Q):
                distance[i][j] = math.sqrt( (i - self.P/2)**2 + (j - self.Q/2)**2 )

        return distance
#Creating the filters-----------------------------------------------------------
    def __createHighFilter(self, distance, delimiter):
        H = np.zeros((self.P, self.Q, self.O), dtype=np.float32)
        for i in range(self.P):
            for j in range(self.Q):
                for k in range(self.O):
                    if distance[i][j] > delimiter:
                        H[i][j][k] = 1

        return H

    def __createLowFilter(self, distance, delimiter):
        H = np.zeros((self.P, self.Q, self.O), dtype=np.float32)
        for i in range(self.P):
            for j in range(self.Q):
                for k in range(self.O):
                    if distance[i][j] < delimiter:
                        H[i][j][k] = 1

        return H

#Applying the selected filter---------------------------------------------------

    def __applyFilter(self, H, DFT):
        #Opencv method for dft return 2 channels (real and imaginary)
        filteredDft = np.zeros((self.P, self.Q, self.O), dtype=np.float32)
        for i in range(self.P):
            for j in range(self.Q):
                for k in range(self.O):
                    filteredDft[i][j][k] =  H[i][j][k]*DFT[i][j][k]

                    #filteredDft[i][j][k][0] =  H[i][j][k]*DFT[i][j][0]
                    #filteredDft[i][j][k][1] =  H[i][j][k]*DFT[i][j][1]

        return filteredDft

#DFT calculation----------------------------------------------------------------
    def __dftCalculation(self, image):

        image = self.__centeringImage(image)

        #return cv2.dft(np.float32(image),flags = cv2.DFT_REAL_OUTPUT)
        return np.fft.fft2(image)

#Inverse DFT calculation--------------------------------------------------------

    def __inverseDft(self, image):

        #image = self.__centeringImage(image)
        #imageBack = cv2.idft(image)
        #imageBack = cv2.magnitude(imageBack[:,:,:,0],imageBack[:,:,:,1])
        imageBack = np.fft.ifft2(image)
        imageBack = self.__centeringImage(imageBack)

        return imageBack

#Returning the image to the original size---------------------------------------

    #TODO: Put the intensity pixels in 0..255
    def __resizeImage(self, image):
        transformedImage = np.zeros((self.M, self.N, self.O), dtype=np.float32)
        bits32 = 2**32
        bits8 = 2**8
        for i in range(self.M):
            for j in range(self.N):
                for k in range(self.O):
                    #value = (image[i][j][k] / bits32) * bits8
                    value = image[i][j][k]
                    #
                    if value > 255:
                        value = 255

                    if value < 0:
                        value = 0

                    transformedImage[i][j][k] = value
        transformedImage = np.uint8(transformedImage)

        return transformedImage

#Opcional method to show results-----------------------------------------------

    def __showResults(self, originalImage, transformedImage):

        transformedImage = np.uint8(transformedImage)
        originalImage = np.uint8(originalImage)
        # plt.figure(1)
        # plt.subplot(111),plt.imshow(originalImage, cmap = 'gray')
        # plt.title('Input image '), plt.xticks([]), plt.yticks([])
        # plt.show()
        # print originalImage.shape
        # print imageBack.shape
        # print transformedImage.shape
        imageComparison = np.hstack((originalImage, transformedImage))
        cv2.imshow('Images Comparison: Original x Filter applied', imageComparison)
        cv2.waitKey(0)

#Main method------------------------------------------------------------------

    def fourierTransform(self, filterChoice, delimiter, results=False):
        distance = self.__distanceCalculation()
        DFT = self.__dftCalculation(self.transformedImage)

        if filterChoice == 0:
            H = self.__createLowFilter(distance, delimiter)
        else:
            H = self.__createHighFilter(distance, delimiter)

        filteredDft = self.__applyFilter(H, DFT)

        imageBack = self.__inverseDft(filteredDft)

        filteredImage = self.__resizeImage(imageBack)

        if results == True:
            self.__showResults(self.originalImage, filteredImage)

        return filteredImage

    def emphasisFilter(self, filterChoice, delimiter, results=False):

        transformedImage = np.zeros(self.originalImage.shape, dtype=np.float32)
        fourierFilter = self.fourierTransform(filterChoice, delimiter, False)

        print fourierFilter
        c = -1

        for i in range(self.originalImage.shape[0]):
            for j in range(self.originalImage.shape[1]):
                for k in range(self.originalImage.shape[2]):
                    value = self.originalImage[i][j][k] + c*fourierFilter[i][j][k]
                    if value < 0:
                        value = 0

                    if value > 255:
                        value = 255

                    transformedImage[i][j][k] = value

        #Converting the images to unsigned int 8 bits
        self.originalImage = np.uint8(self.originalImage)
        fourierFilter = np.uint8(fourierFilter)
        transformedImage = np.uint8(transformedImage)

        if results == True:
            self.__showResults(self.originalImage, fourierFilter, transformedImage)

        return fourierFilter, transformedImage
