import cv2 as cv

# Read the image
img = cv.imread('C:/Users/Robertas/Desktop/P1050494.JPG', 1)

# Resize it down to 10% of original size (naudoju 3872x496 dydzio nuotrauka)
scale_percent = 10
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
img_resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)

# Convert resized image to grayscale
img_gray = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)

# Threshold
ret, thresh = cv.threshold(img_gray, 100, 255, cv.THRESH_BINARY_INV)

# Find contours
contours, hierarchy  = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Draw contours on a white canvas of the resized image
canvas = cv.cvtColor(cv.cvtColor(thresh, cv.COLOR_GRAY2BGR), cv.COLOR_BGR2GRAY) * 0 + 255
img_with_contours = cv.drawContours(canvas, contours, -1, (0, 255, 0), 5)

# Display the result
cv.imshow("Konturas", img_with_contours)

cv.waitKey(0)
cv.destroyAllWindows()