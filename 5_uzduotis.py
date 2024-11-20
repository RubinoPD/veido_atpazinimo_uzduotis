import numpy as np
import cv2
from matplotlib import pyplot as plt

# Įkeliame kairi ir dešini vaizdus
imgL = cv2.imread('faceL.jpg', 0)
imgR = cv2.imread('faceR.jpg', 0)

# Patikriname ir suderiname dydžius
if imgL.shape != imgR.shape:
    imgR = cv2.resize(imgR, (imgL.shape[1], imgL.shape[0]))

# Sukuriame StereoBM objektą
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

# Apskaičiuojame gylio žemėlapį
disparity = stereo.compute(imgL, imgR)

# Atvaizduojame rezultatą
plt.imshow(disparity, 'gray')
plt.title("Disparity Map")
plt.colorbar()
plt.show()