# USAGE
# python otsu_thresholding.py opencv_logo.png

# import the necessary packages
import sys
import cv2

#Get the image
img = sys.argv[1]

# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread(img)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)
cv2.imshow("Image", image)

# apply Otsu's automatic thresholding -- Otsu's method automatically
# determines the best threshold value `T` for us
(T, threshInv) = cv2.threshold(blurred, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imshow("Threshold", threshInv)
print("Otsu's thresholding value: {}".format(T))

# finally, we can visualize only the masked regions in the image
cv2.imshow("Output", cv2.bitwise_and(image, image, mask=threshInv))
cv2.waitKey(0)
