# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

#To not abbreviate big matrices
np.set_printoptions(threshold='nan')

class BruteForce(object):
    'This class will compare pixel by pixel the difference between the test image and the train images '

    def __init__(self, image1, image2):
        self.image1 = image1
        self.image2 = image2

        self.M, self.N, self.O = self.image1.shape

    def __averageImage(self, image):
        #Instead I could use np.average(image)
        M, N, O = image.shape

        sumOfElements = 0
        numberElements = M*N*O

        for i in range(M):
            for j in range(N):
                for k in range(O):
                    sumOfElements += image[i][j][k]

        average = sumOfElements / numberElements

        print average, image.shape
        return average, image.shape

    def bruteForce(self):
        self.__averageImage(self.image1)
