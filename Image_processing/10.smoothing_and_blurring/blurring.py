# USAGE
# python blurring.py big_boss.JPG

# import the necessary packages
import sys
import cv2

#Get the image
img = sys.argv[1]

# load the image, display it, and initialize the list of kernel sizes
image = cv2.imread(img)
cv2.imshow("Original", image)
kernelSizes = [(3, 3), (9, 9), (15, 15)]

# loop over the kernel sizes and apply an "average" blur to the image
for (kX, kY) in kernelSizes:
	blurred = cv2.blur(image, (kX, kY))
	cv2.imshow("Average ({}, {})".format(kX, kY), blurred)
	cv2.waitKey(0)

# close all windows to cleanup the screen
cv2.destroyAllWindows()
cv2.imshow("Original", image)

# loop over the kernel sizes and apply a "Gaussian" blur to the image
for (kX, kY) in kernelSizes:
	blurred = cv2.GaussianBlur(image, (kX, kY), 0)
	cv2.imshow("Gaussian ({}, {})".format(kX, kY), blurred)
	cv2.waitKey(0)

# close all windows to cleanup the screen
cv2.destroyAllWindows()
cv2.imshow("Original", image)

# loop over the kernel sizes and apply a "Median" blur to the image
for k in (3, 9, 15):
	blurred = cv2.medianBlur(image, k)
	cv2.imshow("Median {}".format(k), blurred)
	cv2.waitKey(0)