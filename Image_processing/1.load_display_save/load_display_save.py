# Usage python load_display_save.py

# import the necessary packages
import cv2
 
# load the image and show some basic information on it
image = cv2.imread("bb.JPG")
 
# show the image and wait for a keypress
cv2.imshow("big_boss", image)
cv2.waitKey(0)

# save the image -- OpenCV handles converting file types automatically
cv2.imwrite("big_boss_1.jpg", image)