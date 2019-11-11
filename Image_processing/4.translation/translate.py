# USAGE
# python translate.py minion.jpg

# import the necessary packages


import imutils
import cv2
import sys

img = sys.argv[1]
# load the image and show it
image = cv2.imread(img)
cv2.imshow("Original", image)


shifted = imutils.translate(image, 0, 100)
cv2.imwrite("image.jpg",shifted)
cv2.imshow("Shifted Down", shifted)
cv2.waitKey(0)