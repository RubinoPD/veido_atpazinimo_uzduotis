import cv2 as cv


img = cv.imread('C:/Users/Robertas/Desktop/nuotrauka.jpg', 1)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

white = cv.imread('C:/Users/Robertas/Desktop/nuotrauka.jpg', 0)

#Piešiama konturo atitiktis pagal foto

ret, im = cv.threshold(img_gray, 100, 225, cv.THRESH_BINARY_INV)
contours, hierarchy  = cv.findContours(im, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
img = cv.drawContours(white, contours, -1, (0,255,75), 5)

cv.imshow("Konturas", img)

print(img)

cv.waitKey(0)