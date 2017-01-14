# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

#To not abbreviate big matrices
np.set_printoptions(threshold='nan')

class BruteForce(object):
    'This class will compare pixel by pixel the difference between two images'

    def __init__(self, image1, image2):
        self.image1 = np.float32(image1)
        self.image2 = np.float32(image2)

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

        return average

    def bruteForce(self):
        avg1 = self.__averageImage(self.image1)
        avg2 = self.__averageImage(self.image2)
        A = self.image1
        B = self.image2
        print self.image1.shape, self.image2.shape

#        correlationMatrix =np.zeros((self.M, self.N, self.O), dtype=np.float32)
        # raise Exception("TEXTO")
        numerator    = 0
        denominator1 = 0
        denominator2 = 0
        for i in range(self.M):
            for j in range(self.N):
                for k in range(self.O):
                    A = self.image1[i][j][k]
                    B = self.image2[i][j][k]

                    numerator    += (A - avg1) * (B - avg2)
                    denominator1 += (A - avg1)**2
                    denominator2 += (B - avg2)**2


        correlation = numerator / (denominator1 * denominator2)**0.5

        print "The images are " , correlation * 100, "% equals"
        return correlation
