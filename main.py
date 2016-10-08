import cv2
import numpy as np
import matplotlib.pyplot as plt

originalImage = cv2.imread("source/TrainDatabase/1.jpg", 0)
transformedImage = cv2.imread("source/TrainDatabase/1.jpg", 0)

#originalImage = cv2.imread("pout.tif", 0)
#transformedImage = cv2.imread("pout.tif", 0)

#calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
#The parameters are:
#images: Source images
#channels: Layers of RGB, 0 for  gray scale, 1 or 2 for R,G and B
#mask: If you want a histogram of a full image, pass none, else create a mask image for that and give it as mask
#histSize: Number of bits of the image
#range: Range used to calculate

histr = cv2.calcHist([originalImage],[0],None,[256],[0,256])
#plt.plot(histr)
#plt.show()

N = len(originalImage)
M = len(originalImage[0])
totalPixels = N*M
histLen = len(histr)
print type(histr)
for i in range(histLen):
    histr[i] = histr[i] / totalPixels

#plt.plot(histr)
#plt.show()
L = 256 #Number of pixels
s = np.zeros((histLen)) #Creating a vector with the same length of the histogram, to make a transformation function
initialValue = 1
#Creating the transformation function 's'
s[0] = np.rint( (L - 1) * histr[i] ) #Round to closest int number
for i in range(initialValue, histLen):
    s[i] = np.rint( s[i-1] + (L - 1) * histr[i] )

fig1 = plt.figure(2)
fig1.suptitle('Transformation function',fontsize=14, fontweight='bold')
ax = fig1.add_subplot(111)
ax.set_xlabel('x', fontweight='bold')
ax.set_ylabel('y', fontweight='bold')
plt.plot(s)
plt.show()

for i in range(N):
    for j in range(M):
        #Applying the transformation to the originalImage
        transformedImage[i][j] = s[originalImage[i][j]]


histr = cv2.calcHist([originalImage],[0],None,[256],[0,256])
plt.plot(histr)
plt.show()
histr = cv2.calcHist([transformedImage],[0],None,[256],[0,256])
plt.plot(histr)
plt.show()
cv2.imshow("Transformed Image", transformedImage)
cv2.imshow("Image", originalImage)
cv2.waitKey(0)

equ = cv2.equalizeHist(originalImage)
res = np.hstack((transformedImage,equ)) #stacking images side-by-side
cv2.imwrite('result.png',res)


clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(originalImage)
res = np.hstack((originalImage,cl1))
cv2.imwrite('clahe_2.jpg',res)
