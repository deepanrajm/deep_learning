# USAGE
# python finding_drawing_contours.py images/basic_shapes.png

# import the necessary packages
import numpy as np
import sys
import cv2
import imutils

img = sys.argv[1]

print (img)

# load the image and convert it to grayscale
image = cv2.imread(img)
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# show the original image
cv2.imshow("Original", image)
cv2.waitKey(0)

(T, threshInv) = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imshow("threshInv",threshInv)
cv2.waitKey(0)

# find all contours in the image and draw ALL contours on the image
cnts = cv2.findContours(threshInv.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
clone = image.copy()

for cnt in cnts:
	x,y,w,h = cv2.boundingRect(cnt)
	
	if (w<120 and w>90 and h<70 and h>30):
		print (x,y,w,h)
		cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
		cv2.imshow("output", image)
		cv2.waitKey(0)


#cv2.drawContours(clone, cnts, -1, (0, 255, 0), 2)
#print("Found {} contours".format(len(cnts)))

# show the output image
#cv2.imshow("All Contours", clone)
#cv2.waitKey(0)

# re-clone the image and close all open windows
clone = image.copy()
cv2.destroyAllWindows()



# find contours in the image, but this time keep only the EXTERNAL
# contours in the image



# cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if imutils.is_cv2() else cnts[1]
# cv2.drawContours(clone, cnts, -1, (0, 255, 0), 2)
# print("Found {} EXTERNAL contours".format(len(cnts)))

# # show the output image
# cv2.imshow("All Contours", clone)
# cv2.waitKey(0)

# # re-clone the image and close all open windows
# clone = image.copy()
# cv2.destroyAllWindows()

# # loop over the contours individually
# for c in cnts:
# 	# construct a mask by drawing only the current contour
# 	mask = np.zeros(gray.shape, dtype="uint8")
# 	cv2.drawContours(mask, [c], -1, 255, -1)

# 	# show the images
# 	cv2.imshow("Image", image)
# 	cv2.imshow("Mask", mask)
# 	cv2.imshow("Image + Mask", cv2.bitwise_and(image, image, mask=mask))
# 	cv2.waitKey(0)
