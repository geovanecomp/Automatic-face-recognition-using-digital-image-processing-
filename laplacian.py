# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# a = np.array([[1,2,3],[2,3,4]])
# a = np.insert(a, 0, values=0, axis=1)
#
# print a
#
# a = np.delete(a,1, 0)
#
# print a


#Para nao 'abreviar' matrizes grandes
np.set_printoptions(threshold='nan')

#1º Step: Get the images and define auxiliary matrices--------------------------

#originalImage = cv2.imread('source/TrainDatabase/1.jpg', 0)
#transformedImage = cv2.imread('source/TrainDatabase/1.jpg', 0)

originalImage = cv2.imread('lua.tif', 0)
lenImgX, lenImgY = np.shape(originalImage) #Number of lines and columns

#transformedImage = np.copy(originalImage)
#transformedImage = np.zeros((lenImgX, lenImgY), dtype=np.uint8)

mask = np.zeros((3,3), dtype=np.int8)
lenMaskX, lenMaskY = np.shape(mask)

#-------------------------------------------------------------------------------
#2º Step: Define the functions--------------------------------------------------

def fillMask():

    dimension = 3
    mask = np.ones((dimension, dimension), dtype=np.int8)
    center = dimension / 2
    mask[center][center] = -8
    return mask

def addRows(matrix, quantity, position, values=0):
    for i in range(quantity):
        matrix = np.insert(matrix, position, values, axis=0)
    return matrix

def addColumns(matrix, quantity, position, values=0):
    for i in range(quantity):
        matrix = np.insert(matrix, position, values, axis=1)
    return matrix

def deleteRows(matrix, quantity, position):
    for i in range(quantity):
        position -= 1
        matrix = np.delete(matrix, position, axis=0)
    return matrix

def deleteColumns(matrix, quantity, position):
    for i in range(quantity):
        position -= 1
        matrix = np.delete(matrix, position, axis=1)
    return matrix


def laplacianMask(f):
    f = np.int16(f)
    lenX, lenY = np.shape(f)
    border = 1
    #g = np.zeros((lenX, lenY), dtype=np.uint8)
    g = np.zeros((lenX, lenY))
    for i in range(border, lenX - border):
        for j in range (border, lenY - border):
            # diagonalValues = f[i+1][j+1] + f[i-1][j+1] + f[i-1][j-1]+ f[i+1][j-1]
            # mainValues = f[i+1][j] + f[i-1][j] + f[i][j+1] + f[i][j-1]
            # centerValue = 8 * f[i][j]
            #
            # g[i][j] = mainValues + diagonalValues - centerValue


            value = f[i+1][j] + f[i-1][j] + f[i][j+1] + f[i][j-1] + f[i+1][j+1] + f[i-1][j+1] + f[i-1][j-1] + f[i+1][j-1] - 8 * f[i][j]
            #value = - f[i+1][j] - f[i-1][j] - f[i][j+1] - f[i][j-1] - f[i+1][j+1] - f[i-1][j+1] - f[i-1][j-1] - f[i+1][j-1] + 8 * f[i][j]
            #value = f[i+1][j] + f[i-1][j] + f[i][j+1] + f[i][j-1] -4 * f[i][j]

            if value > 255:
                value = 255

            if value < 0:
                value = 0

            g[i][j] = value

    g = np.uint8(g)
    return g

def applyMask(image):
    image = np.int16(image)
    w = fillMask()
    m, n = w.shape

    image = addRows(image, 2, 0)
    image = addRows(image, 2, len(image))

    image = addColumns(image, 2, 0)
    image = addColumns(image, 2, len(image[0]))

    #g = np.zeros(image.shape, dtype=np.uint8)
    g = np.zeros(image.shape, dtype=np.int16)

    a = (m-1)/2
    b = (n-1)/2

    # for x in range(len(image)-1):
    #     for y in range(len(image[0])-1):
    #         for s in range(-a, a):
    #             for t in range(-b, b):
    #                 g[x+1][y+1] = w[s][t]*image[x+s][y+t] + g[x+1][y+1]

    for x in range(len(image)-1):
        for y in range(len(image[0])-1):
            for s in range(2):
                for t in range(2):
                    value = w[s][t]*image[x+s-1][y+t-1] + g[x+1][y+1]

                    # if value < 0:
                    #     value = 0
                    #
                    # if value > 255:
                    #     value = 255

                    g[x+1][y+1] = value

                    #g[x+1][y+1] = w[s][t]*image[x+s-1][y+t-1] + g[x+1][y+1]
            # if g[x+1][y+1] > 255:
            #     g[x+1][y+1] = 255
            #
            # if g[x+1][y+1] < 0:
            #     g[x+1][y+1] = 0

    g = deleteRows(g, 2, 0)
    g = deleteRows(g, 2, len(g))

    g = deleteColumns(g, 2, 0)
    g = deleteColumns(g, 2, len(g[0]))

    g = np.uint8(g)

    return g


def laplacianFilter(image):
    #image = np.int16(image)
    transformedImage = np.zeros(image.shape, dtype=np.uint8)
    laplacian = laplacianMask(image)
    #laplacian = applyMask(image)

    c = -1
    for i in range(len(image)):
        for j in range(len(image[0])):
            value = image[i][j] + c*laplacian[i][j]
            if value < 0:
                value = 0

            if value > 255:
                value = 255
            transformedImage[i][j] = value

    imageComparison = np.hstack((image, laplacian, transformedImage))
    cv2.imshow('Laplacian', imageComparison)
    cv2.waitKey(0)

    return transformedImage


transformedImage = laplacianFilter(originalImage)
#transformedImage = applyMask(originalImage)

# imageComparison = np.hstack((originalImage, transformedImage))
#
# cv2.imshow('Images Comparison: Original x Transformed', imageComparison)
# cv2.waitKey(0)
