from keras.models import load_model
from keras.preprocessing import image
from pathlib import Path
import numpy as np

# Load the model we trained
model = load_model('bird_model.h5')

for f in sorted(Path(".").glob("*.png")):

    # Load an image file to test
    image_to_test = image.load_img(str(f), target_size=(32, 32))

    # Convert the image data to a numpy array suitable for Keras
    image_to_test = image.img_to_array(image_to_test)

    # Normalize the image the same way we normalized the training data (divide all numbers by 255)
    image_to_test /= 255

    # Add a fourth dimension to the image since Keras expects a list of images
    list_of_images = np.expand_dims(image_to_test, axis=0)

    # Make a prediction using the bird model
    results = model.predict(list_of_images)

    # Since we only passed in one test image, we can just check the first result directly.
    image_likelihood = results[0][0]

    # The result will be a number from 0.0 to 1.0 representing the likelihood that this image is a bird.
    if image_likelihood > 0.5:
        print(f, "is most likely a bird!", image_likelihood)
    else:
        print(f, "is most likely NOT a bird! ",image_likelihood)
