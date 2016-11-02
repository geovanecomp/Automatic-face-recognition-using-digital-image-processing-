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
