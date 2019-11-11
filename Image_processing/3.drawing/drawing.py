# USAGE
# python drawing.py

# import the necessary packages
import numpy as np
import cv2

# initialize our canvas as a 500x500 with 3 channels, Red, Green,
# and Blue, with a White background
canvas = np.ones((500, 500, 3), dtype="uint8")*255


# draw a green line from the top-left corner of our canvas to the
# bottom-right
black = (0, 0, 0)
cv2.line(canvas, (0, 0), (400, 400), black,3)
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

red =[0,0,255]

# draw  rectangle, this time we'll make it red and 5 pixels thick
cv2.rectangle(canvas, (150, 100), (100, 225), red, 5)
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

# let's draw one last rectangle: green and filled in
green = (0, 255, 0)
cv2.rectangle(canvas, (300, 50), (425, 125), green, -1)
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)


canvas = np.zeros((500, 500, 3), dtype="uint8")


# let's go crazy and draw 25 random circles
for i in range(0, 25):
	# randomly generate a radius size between 5 and 200, generate a random
	# color, and then pick a random point on our canvas where the circle
	# will be drawn
	radius = np.random.randint(5, high=300)
	color = np.random.randint(0, high=256, size = (3,)).tolist()
	pt = np.random.randint(0, high=400, size = (2,))

	# draw our random circle
	cv2.circle(canvas, tuple(pt), radius, color, -1)

# Show our masterpiece
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

# load the image of big boos 
image = cv2.imread("big_boss.jpg")

cv2.putText(image, 'Big Boss 2', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

cv2.imshow("Ouput", image)
cv2.waitKey(0)

# draw a circle around my face, two filled in circles covering my eyes, and
# a rectangle surrounding my mouth
cv2.circle(image, (315, 106), 70, (0, 0, 255), 2)
cv2.circle(image, (315, 105), 10, (0, 0, 255), -1)
cv2.rectangle(image, (135, 30), (505, 190), (0, 0, 255), 3)

# show the output image
cv2.imshow("Ouput", image)
cv2.waitKey(0)
