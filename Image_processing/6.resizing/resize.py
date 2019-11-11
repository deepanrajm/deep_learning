# USAGE
# python resize.py minion.jpg

# import the necessary packages
import imutils
import cv2
import sys

# Get the image
img = sys.argv[1]

# load the image and show it
image = cv2.imread(img)
cv2.imshow("Original", image)

crop = image[10:300,10:300]
cv2.imshow("corp",crop)
cv2.waitKey(0)


resized = cv2.resize(image, (500,500))
cv2.imshow("Resized via Function", resized)
cv2.waitKey(0)

resized = imutils.resize(image, width=500)
cv2.imshow("Resized via Function", resized)
cv2.waitKey(0)

for i in range (100,1000,100):

	resized = imutils.resize(image, width=i)
	cv2.imshow("Resized images", resized)
	cv2.waitKey(0)
