# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#To not abbreviate big matrices
np.set_printoptions(threshold='nan')

class HistogramEqualization(object):
    'This class is responsible for apply the histogram equalization of an image - Distribute the intensity pixels evenly'
    def __init__(self, image):
        self.originalImage = image
        self.transformedImage = np.zeros(self.originalImage.shape, dtype=np.uint8)
        self.M, self.N, self.O = self.originalImage.shape

#1º Step: Normalize the original image's histogram------------------------------
    def __normalizeHistogram(self, histogram):

        #Getting importants informations
        normalizedHistogram = np.copy(histogram)
        totalPixels = self.M*self.N
        histLen = len(normalizedHistogram)

        #Normalizing the histogram (The sum of the all elements will be 1)
        for i in range(histLen):
            normalizedHistogram[i] = histogram[i] / totalPixels

        return normalizedHistogram


#2º Step: Create a transformation function--------------------------------------
    def __makeTransformationFunction(self, histogram):

        L = 256 #Number of pixels
        histLen = len(histogram)
        s = np.zeros((histLen)) #Creating a vector with the same length of the histogram
        initialValue = 1

        #Creating the transformation function 's'
        s[0] = np.rint( (L - 1) * histogram[0] ) #Round to closest int number
        for i in range(initialValue, histLen):
            s[i] = np.rint( s[i-1] + (L - 1) * histogram[i] )

        return s

#3º Step: Apply the transformation----------------------------------------------
    def __applyTransformation(self, s):

        for i in range(self.M):
            for j in range(self.N):
                for k in range(self.O):
                    #Applying the transformation to the originalImage
                    self.transformedImage[i][j][k] = s[self.originalImage[i][j][k]]

#Opcional Step: Show the results------------------------------------------------------
    def __showResults(self, histrOriginalImage, histrTransformedImage, s):

        fig1 = plt.figure(1)
        fig1.suptitle('Histogram Comparison', fontsize=14, fontweight='bold')

        ax = fig1.add_subplot(1,1,1)
        ax.set_xlabel('Input Intensity', fontweight='bold')
        ax.set_ylabel('Output Intensity', fontweight='bold')

        blue_patch = mpatches.Patch(color='blue', label='Original Histogram')
        red_patch = mpatches.Patch(color='red', label='Transformed Histogram')

        plt.legend(handles=[blue_patch, red_patch])

        plt.plot(histrOriginalImage, color='blue')
        plt.plot(histrTransformedImage, color='red')

        plt.show()

        #Plot of the transformation function
        fig2 = plt.figure(2)
        fig2.suptitle('Transformation function',fontsize=14, fontweight='bold')
        ax = fig2.add_subplot(111)
        ax.set_xlabel('x', fontweight='bold')
        ax.set_ylabel('y', fontweight='bold')
        plt.plot(s)
        plt.show()

        imageComparison = np.hstack((self.originalImage, self.transformedImage))
        cv2.imshow('Images Comparison: Original x My Hist Equ', imageComparison)
        cv2.waitKey(0)

#Final Step --------------------------------------------------------------------
    #Calculate the histogram. If results = true, will show the plots
    def calculate(self, results=False):

        #calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
        #The parameters are:
        #images: Source images
        #channels: Layers of RGB, 0 for  gray scale, 1 or 2 for R,G and B
        #mask: If you want a histogram of a full image, pass none, else create a mask image for that and give it as mask
        #histSize: Number of bits of the image
        #range: Range used to calculate
        histrOriginalImage = cv2.calcHist([self.originalImage],[self.O - 1],None,[256],[0,256])

        normalizedHistrOriginalImage = self.__normalizeHistogram(histrOriginalImage)
        s = self.__makeTransformationFunction(normalizedHistrOriginalImage)
        self.__applyTransformation(s)

        if results == True:
            histrTransformedImage = cv2.calcHist([self.transformedImage],[self.O - 1],None,[256],[0,256])
            self.__showResults(histrOriginalImage, histrTransformedImage, s)

        return self.transformedImage

#Bonus: ------------------------------------------------------------------------
#Otimization for histogram equalization
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# cl1 = clahe.apply(originalImage)
# res = np.hstack((originalImage,cl1))
# cv2.imwrite('clahe_2.jpg',res)
