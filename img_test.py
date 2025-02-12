import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import imutils
from funciones import *

img = cv.imread("img/red_green.jpg")
assert img is not None

img_h, img_w, _ = img.shape

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)


mask_red = [[180-20, 120, 40], [180, 255, 255]]
mask_green = [[60-20, 120, 40], [60+20, 255, 255]]


ROI = [0, 0, img_w, img_h]
red_cnts = find_contours(hsv, mask_red, ROI)
green_cnts = find_contours(hsv, mask_green, ROI)



cv.drawContours(img, red_cnts, -1, (0,255,0), 2)
cv.drawContours(img, green_cnts, -1, (0,255,0), 2)




img = imutils.resize(img, 400)

#cv.imshow("mask", mask)
cv.imshow("original", img)

cv.waitKey(0)
cv.destroyAllWindows()

