# USAGE
# python masking.py minion.jpg

# import the necessary packages
import numpy as np
import sys
import cv2


img = sys.argv[1]

# load the image and display it it
image = cv2.imread(img)
cv2.imshow("Original", image)



# Create a white rectangle

mask = np.zeros(image.shape[:2], dtype="uint8")
cv2.rectangle(mask, (452, 243), (690, 590), 255, -1)
cv2.imshow("Mask", mask)

# Apply out mask -- notice 
masked = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Mask Applied to Image", masked)
cv2.waitKey(0)

# Now, let's make a circular mask 

mask = np.zeros(image.shape[:2], dtype="uint8")
cv2.circle(mask, (585, 98), 50, 255, -1)
masked = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Mask", mask)
cv2.imshow("Mask Applied to Image", masked)
cv2.waitKey(0)