from sklearn.metrics import confusion_matrix, classification_report
from keras.datasets import cifar10
from keras.models import load_model

# Load data set
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# CIFAR10 has ten types of images labeled from 0 to 9. We only care about birds, which are labeled as class #2.
# So we'll cheat and change the Y values. Instead of each class being labeled from 0 to 9, we'll set it to True
# if it's a bird and False if it's not a bird.
y_test = y_test == 2

# Normalize image data (pixel values from 0 to 255) to the 0-to-1 range
x_test = x_test.astype('float32')
x_test /= 255

# Load the model we trained
model = load_model('bird_model.h5')
predictions = model.predict(x_test, batch_size=32, verbose=1)

# If the model is more than 50% sure the object is a bird, call it a bird.
# Otherwise, call it "not a bird".
predictions = predictions > 0.5

# Calculate how many mis-classifications the model makes
tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
print("True Positives: ",tp)
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)

# Calculate Precision and Recall for each class
report = classification_report(y_test, predictions)
print(report)