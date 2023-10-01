import sys
import cv2
import numpy as np
import pygame
import time
from matplotlib import pyplot as plt

print(str(sys.argv[1]))
im = cv2.imread(str(sys.argv[1]))
cv2.waitKey()
gray1 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
cv2.imwrite(str(sys.argv[1]), gray1)

# CONTOUR DETECTION CODE
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)

contours1, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
contours2, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img2 = im.copy()
out = cv2.drawContours(img2, contours2, -1, (250, 250, 250), 1)

plt.subplot(331), plt.imshow(im), plt.title('GRAY')
plt.xticks([]), plt.yticks([])

img = cv2.imread(str(sys.argv[1]), 0)
ret, thresh = cv2.threshold(img, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, 1, 2)
cnt = contours[0]

M = cv2.moments(cnt)
perimeter = cv2.arcLength(cnt, True)
area = cv2.contourArea(cnt)
epsilon = 0.1 * cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)

contour_list = []
pothole_detected = False

for c in contours:
    rect = cv2.boundingRect(c)
    approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
    area = cv2.contourArea(c)
    print(str(rect[2]) + "========" + str(rect[3]) + "------------------------")
    
    if rect[2] < 20 or rect[3] < 10:
        continue
    
    print("Pothole Detected")
    pothole_detected = True
    
    contour_list.append(c)
    x, y, w, h = rect
    cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 8)
    cv2.putText(img2, 'POTHOLE Detected', (x + w + 40, y + h), 0, 2.0, (0, 255, 0))
    cv2.drawContours(img2, contour_list, -1, (255, 0, 0), 2)
    cv2.imshow('Objects Detected', img2)
    cv2.waitKey(0)
    pygame.init()
    pygame.mixer.music.load("buzz.mp3")
    pygame.mixer.music.play()
    time.sleep(5)

if not pothole_detected:
    print("No Pothole Detected")

k = cv2.isContourConvex(cnt)

# Image Processing
blur = cv2.blur(im, (5, 5))
gblur = cv2.GaussianBlur(im, (5, 5), 0)
median = cv2.medianBlur(im, 5)
erosion = cv2.erode(median, np.ones((5, 5), np.uint8), iterations=1)
dilation = cv2.dilate(erosion, np.ones((5, 5), np.uint8), iterations=5)
closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
edges = cv2.Canny(dilation, 9, 220)

# Plotting using matplotlib
plt.subplot(332), plt.imshow(blur), plt.title('BLURRED')
plt.xticks([]), plt.yticks([])
plt.subplot(333), plt.imshow(gblur), plt.title('Gaussian blur')
plt.xticks([]), plt.yticks([])
plt.subplot(334), plt.imshow(median), plt.title('Median blur')
plt.xticks([]), plt.yticks([])
plt.subplot(337), plt.imshow(img, cmap='gray')
plt.title('Dilated Image'), plt.xticks([]), plt.yticks([])
plt.subplot(338), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(335), plt.imshow(erosion), plt.title('EROSION')
plt.xticks([]), plt.yticks([])
plt.subplot(336), plt.imshow(closing), plt.title('Closing')
plt.xticks([]), plt.yticks([])

plt.show()
