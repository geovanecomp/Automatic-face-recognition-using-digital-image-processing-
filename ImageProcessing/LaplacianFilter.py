# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#To not abbreviate big matrices
np.set_printoptions(threshold='nan')

class LaplacianFilter(object):
    'This class is responsible for apply the Laplacian Filter on an image - Emphasizes details'

    def __init__(self, image):
        self.originalImage = np.float32(image)

#Define the auxiliary functions-------------------------------------------------

    def __fillMask(self):

        dimension = 3
        mask = np.ones((dimension, dimension), dtype=np.int8)
        center = dimension / 2
        mask[center][center] = -8
        return mask

    def __addRows(self, matrix, quantity, position, values=0):
        for i in range(quantity):
            matrix = np.insert(matrix, position, values, axis=0)
        return matrix

    def __addColumns(self, matrix, quantity, position, values=0):
        for i in range(quantity):
            matrix = np.insert(matrix, position, values, axis=1)
        return matrix

    def __deleteRows(self, matrix, quantity, position):
        for i in range(quantity):
            position -= 1
            matrix = np.delete(matrix, position, axis=0)
        return matrix

    def __deleteColumns(self, matrix, quantity, position):
        for i in range(quantity):
            position -= 1
            matrix = np.delete(matrix, position, axis=1)
        return matrix

    def __showResults(self, laplacian, transformedImage):
        imageComparison = np.hstack((self.originalImage, laplacian, transformedImage))
        cv2.imshow('Laplacian Filter: Original x Laplacian x Transformed Image', imageComparison)
        cv2.waitKey(0)

        plt.figure(1)
        plt.subplot(131),plt.imshow(self.originalImage, cmap = 'gray')
        plt.title('Input image '), plt.xticks([]), plt.yticks([])
        plt.subplot(132),plt.imshow(laplacian, cmap = 'gray')
        plt.title('Mask'), plt.xticks([]), plt.yticks([])
        plt.subplot(133),plt.imshow(transformedImage, cmap = 'gray')
        plt.title('Filter applied'), plt.xticks([]), plt.yticks([])
        plt.show()

#Using the derivative approximation---------------------------------------------

    def __laplacianMask(self, f):
        lenX, lenY, lenDim = np.shape(f)

        border = 1
        g = np.zeros((lenX, lenY, lenDim))
        for i in range(border, lenX - border):
            for j in range (border, lenY - border):
                for k in range(lenDim):
                    diagonalValues = f[i+1][j+1][k] + f[i-1][j+1][k] + f[i-1][j-1][k] + f[i+1][j-1][k]
                    mainValues = f[i+1][j][k] + f[i-1][j][k] + f[i][j+1][k] + f[i][j-1][k]
                    centerValue = 8 * f[i][j][k]

                    value = mainValues + diagonalValues - centerValue

                    if value > 255:
                        value = 255

                    if value < 0:
                        value = 0

                    g[i][j][k] = value
        print g.shape
        return g

#Using the neighborhood method -------------------------------------------------

    def __applyMask(self, image):
        w = self.__fillMask()
        m, n = w.shape

        image = self.__addRows(image, 2, 0)
        image = self.__addRows(image, 2, len(image))

        image = self.__addColumns(image, 2, 0)
        image = self.__addColumns(image, 2, len(image[0]))

        M, N, O = image.shape
        g = np.zeros(image.shape, dtype=np.float32)
        a = (m-1)/2
        b = (n-1)/2
        value = 0

        for x in range(M-2):
            for y in range(N-2):
                for k in range(O):
                    for s in range(len(w)):
                        for t in range(len(w[0])):
                            value = w[s][t]*image[x+s][y+t][k] + value

                    if value < 0:
                        value = 0

                    if value > 255:
                        value = 255

                    g[x+1][y+1][k] = value
                    value = 0

        g = self.__deleteRows(g, 2, 0)
        g = self.__deleteRows(g, 2, len(g))

        g = self.__deleteColumns(g, 2, 0)
        g = self.__deleteColumns(g, 2, len(g[0]))

        return g

#Main method--------------------------------------------------------------------

    def laplacianFilter(self, results=False):

        transformedImage = np.zeros(self.originalImage.shape, dtype=np.float32)
        M, N, O = np.shape(self.originalImage)

        #I have made the laplacian filter using two forms
        laplacian = self.__laplacianMask(self.originalImage)
        #laplacian = self.__applyMask(self.originalImage)

        c = -1

        for i in range(M):
            for j in range(N):
                for k in range(O):
                    value = self.originalImage[i][j][k] + c*laplacian[i][j][k]

                    if value < 0:
                        value = 0

                    if value > 255:
                        value = 255

                    transformedImage[i][j][k] = value

        #Converting the images to unsigned int 8 bits
        self.originalImage = np.uint8(self.originalImage)
        laplacian = np.uint8(laplacian)
        transformedImage = np.uint8(transformedImage)

        if results == True:
            self.__showResults(laplacian, transformedImage)

        return laplacian, transformedImage
