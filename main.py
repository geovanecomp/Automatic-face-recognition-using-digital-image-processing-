# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#1º Step: Get the images-------------------------------------------------------------

#originalImage = cv2.imread('source/TrainDatabase/1.jpg', 0)
#transformedImage = cv2.imread('source/TrainDatabase/1.jpg', 0)

originalImage = cv2.imread('pout.tif', 0)
transformedImage = cv2.imread('pout.tif', 0)

#2º Step: Normalize-------------------------------------------------------------

#calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
#The parameters are:
#images: Source images
#channels: Layers of RGB, 0 for  gray scale, 1 or 2 for R,G and B
#mask: If you want a histogram of a full image, pass none, else create a mask image for that and give it as mask
#histSize: Number of bits of the image
#range: Range used to calculate

#Plot of the histogram of the original image
histrOriginalImage = cv2.calcHist([originalImage],[0],None,[256],[0,256])
fig1 = plt.figure(1)
fig1.suptitle('Histogram Comparison', fontsize=14, fontweight='bold')
ax = fig1.add_subplot(1,1,1)
#plt.ylabel('teste', fontweight='bold')
ax.set_xlabel('x', fontweight='bold')
ax.set_ylabel('y', fontweight='bold')
blue_patch = mpatches.Patch(color='blue', label='Original Histogram')
red_patch = mpatches.Patch(color='red', label='Transformed Histogram')
plt.legend(handles=[blue_patch, red_patch])
plt.plot(histrOriginalImage, color='blue')

#Getting importants informations
N = len(originalImage)
M = len(originalImage[0])
totalPixels = N*M
histLen = len(histrOriginalImage)

#Normalizing the histogram (The sum of the all elements will be 1)
for i in range(histLen):
    histrOriginalImage[i] = histrOriginalImage[i] / totalPixels

#3º Step: Create a transformation function--------------------------------------

L = 256 #Number of pixels
s = np.zeros((histLen)) #Creating a vector with the same length of the histogram
initialValue = 1

#Creating the transformation function 's'
s[0] = np.rint( (L - 1) * histrOriginalImage[0] ) #Round to closest int number
for i in range(initialValue, histLen):
    s[i] = np.rint( s[i-1] + (L - 1) * histrOriginalImage[i] )

#4º Step: Apply the transformation----------------------------------------------

for i in range(N):
    for j in range(M):
        #Applying the transformation to the originalImage
        transformedImage[i][j] = s[originalImage[i][j]]

#5º Step: Show the results------------------------------------------------------

#Plot of the histogram of the transformed image
histrTransformedImage = cv2.calcHist([transformedImage],[0],None,[256],[0,256])
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

#Opencv histogram equalization
equOpencvImage = cv2.equalizeHist(originalImage)

#Showing the images
#imageComparison = np.concatenate((originalImage, transformedImage), axis=0) #another way to compare images
imageComparison = np.hstack((originalImage, transformedImage, equOpencvImage))
cv2.imshow('Images Comparison: Original x My Hist Equ x Opencv Hist Equ ', imageComparison)
cv2.waitKey(0)

#-------------------------------------------------------------------------------

#Otimization for histogram equalization
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# cl1 = clahe.apply(originalImage)
# res = np.hstack((originalImage,cl1))
# cv2.imwrite('clahe_2.jpg',res)
