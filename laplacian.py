# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#To not abbreviate big matrices
np.set_printoptions(threshold='nan')

#1º Step: Get the images and define auxiliary matrices--------------------------

#originalImage = cv2.imread('source/TrainDatabase/1.jpg', 0)
originalImage = cv2.imread('lua.tif', 0)

#Converting the image to make operations
originalImage = np.float32(originalImage)

lenImgX, lenImgY = np.shape(originalImage) #Number of lines and columns

transformedImage = np.copy(originalImage)

mask = np.zeros((3,3), dtype=np.int8)
lenMaskX, lenMaskY = np.shape(mask)

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
    lenX, lenY = np.shape(f)
    border = 1
    g = np.zeros((lenX, lenY))
    for i in range(border, lenX - border):
        for j in range (border, lenY - border):
            diagonalValues = f[i+1][j+1] + f[i-1][j+1] + f[i-1][j-1]+ f[i+1][j-1]
            mainValues = f[i+1][j] + f[i-1][j] + f[i][j+1] + f[i][j-1]
            centerValue = 8 * f[i][j]

            value = mainValues + diagonalValues - centerValue

            if value > 255:
                value = 255

            if value < 0:
                value = 0

            g[i][j] = value
    return g

def applyMask(image):
    w = fillMask()
    m, n = w.shape

    image = addRows(image, 2, 0)
    image = addRows(image, 2, len(image))

    image = addColumns(image, 2, 0)
    image = addColumns(image, 2, len(image[0]))

    g = np.zeros(image.shape, dtype=np.float32)
    a = (m-1)/2
    b = (n-1)/2
    value = 0

    for x in range(len(image)-2):
        for y in range(len(image[0])-2):
            for s in range(len(w)):
                for t in range(len(w[0])):
                    value = w[s][t]*image[x+s][y+t] + value

            if value < 0:
                value = 0

            if value > 255:
                value = 255

            g[x+1][y+1] = value
            value = 0

    g = deleteRows(g, 2, 0)
    g = deleteRows(g, 2, len(g))

    g = deleteColumns(g, 2, 0)
    g = deleteColumns(g, 2, len(g[0]))

    return g


def laplacianFilter(image):

    transformedImage = np.zeros(image.shape, dtype=np.float32)
    #I have made the laplacian filter using two forms
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


    return laplacian, transformedImage

#3º step: Showing the results---------------------------------------------------

laplacian, transformedImage = laplacianFilter(originalImage)

#Converting the images to unsigned int
originalImage = np.uint8(originalImage)
laplacian = np.uint8(laplacian)
transformedImage = np.uint8(transformedImage)

imageComparison = np.hstack((originalImage, laplacian, transformedImage))
cv2.imshow('Laplacian', imageComparison)
cv2.waitKey(0)
