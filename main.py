import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("source/TrainDatabase/1.jpg", 0)
img2 = cv2.imread("source/TrainDatabase/1.jpg", 0)

#img = cv2.imread("pout.tif", 0)
#img2 = cv2.imread("pout.tif", 0)


histr = cv2.calcHist([img],[0],None,[256],[0,256])
#plt.plot(histr)
#plt.show()

N = len(img)
M = len(img[0])
totalPixels = N*M
tamHist = len(histr)
print type(histr)
for i in range(tamHist):
    histr[i] = histr[i] / totalPixels

#plt.plot(histr)
#plt.show()
L = 256
s = np.zeros((tamHist))

for i in range(tamHist):
    if i == 0:
        s[i] = (L - 1) * histr[i]
    else:
        s[i] = s[i-1] + (L - 1) * histr[i]

#plt.plot(s)
#plt.show()

for i in range(N):
    for j in range(M):
            img2[i][j] = s[img[i][j]]


histr = cv2.calcHist([img],[0],None,[256],[0,256])
plt.plot(histr)
plt.show()
histr = cv2.calcHist([img2],[0],None,[256],[0,256])
plt.plot(histr)
plt.show()
cv2.imshow("Image transformada", img2)
cv2.imshow("Image", img)
cv2.waitKey(0)

equ = cv2.equalizeHist(img)
res = np.hstack((img2,equ)) #stacking images side-by-side
cv2.imwrite('resultado.png',res)


clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)
res = np.hstack((img,cl1))
cv2.imwrite('clahe_2.jpg',res)
