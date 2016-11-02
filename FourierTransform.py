# -*- coding: UTF-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

#To not abbreviate big matrices
np.set_printoptions(threshold='nan')

class FourierTransform(object):
    def __init__(self, image):
        self.originalImage = np.float32(image)
        self.M, self.N = np.shape(self.originalImage) #Number of lines and columns
        self.P = 2 * self.M
        self.Q = 2 * self.N
        self.transformedImage = np.zeros((self.P, self.Q), dtype=np.float32)
        for i in range(self.M):
            for j in range(self.N):
                self.transformedImage[i][j] = self.originalImage[i][j]

#To move the information in the corners to the center in the frequency domain---
    def __centeringImage(self, image):

        for i in range(self.P):
            for j in range(self.Q):
                image[i][j] = image[i][j] * (-1)**(i+j)

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
        H = np.zeros((self.P, self.Q), dtype=np.float32)
        for i in range(self.P):
            for j in range(self.Q):

                if distance[i][j] > delimiter:
                    H[i][j] = 1

        return H

    def __createLowFilter(self, distance, delimiter):
        H = np.zeros((self.P, self.Q), dtype=np.float32)
        for i in range(self.P):
            for j in range(self.Q):

                if distance[i][j] < delimiter:
                    H[i][j] = 1

        return H

#Applying the selected filter---------------------------------------------------

    def __applyFilter(self, H, DFT):
        #Opencv method for dft return 2 channels (real and imaginary)
        filteredDft = np.zeros((self.P, self.Q, 2), dtype=np.float32)
        for i in range(self.P):
            for j in range(self.Q):
                filteredDft[i][j][0] =  H[i][j]*DFT[i][j][0]
                filteredDft[i][j][1] =  H[i][j]*DFT[i][j][1]

        return filteredDft

#DFT calculation----------------------------------------------------------------
    def __dftCalculation(self, image):

        image = self.__centeringImage(image)

        return cv2.dft(np.float32(image),flags = cv2.DFT_COMPLEX_OUTPUT)

#Inverse DFT calculation--------------------------------------------------------

    def __inverseDft(self, image):

        #image = self.__centeringImage(image)
        imageBack = cv2.idft(image)
        imageBack = self.__centeringImage(imageBack)
        imageBack = cv2.magnitude(imageBack[:,:,0],imageBack[:,:,1])

        return imageBack

#Returning the image to the original size---------------------------------------

    #TODO: Put the intensity pixels in 0..255
    def __resizeImage(self, image):
        transformedImage = np.zeros((self.M, self.N), dtype=np.float32)
        for i in range(self.M):
            for j in range(self.N):
                value = image[i][j]
                # if value > 255:
                #     value = 255
                #
                # if value < 0:
                #     value = 0

                transformedImage[i][j] = value
        #transformedImage = np.uint8(transformedImage)
        return transformedImage

#Opcional method to show results-----------------------------------------------

    def __showResults(self, originalImage, imageBack, transformedImage):
        plt.figure(1)
        plt.subplot(131),plt.imshow(originalImage, cmap = 'gray')
        plt.title('Input image '), plt.xticks([]), plt.yticks([])
        plt.subplot(132),plt.imshow(imageBack, cmap = 'gray')
        plt.title('Mask'), plt.xticks([]), plt.yticks([])
        plt.subplot(133),plt.imshow(transformedImage, cmap = 'gray')
        plt.title('Filter applied'), plt.xticks([]), plt.yticks([])
        plt.show()

#Main method------------------------------------------------------------------

    def FourierTransform(self, filterChoice, delimiter, results=False):
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
            self.__showResults(self.originalImage, imageBack, filteredImage)

        return filteredImage
