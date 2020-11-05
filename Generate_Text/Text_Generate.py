from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import LambdaCallback
import numpy as np
from pathlib import Path


# Load the training data
source_data = Path("sherlock_holmes.txt").read_text()
text = source_data.lower()

# Create a list of unique characters in the data
chars = sorted(list(set(text)))
chars_to_numbers = dict((c, i) for i, c in enumerate(chars))
numbers_to_chars = dict((i, c) for i, c in enumerate(chars))

# Split the text into 40-character sequences
sequence_length = 40

# Capture both each 40-character sequence and the 41-st character that we want to predict
training_sequences = []
training_sequences_next_character = []

# Loop over training text, skipping 40 characters forward on each loop
for i in range(0, len(text) - sequence_length, 40):
    # Grab the 40-character sequence as the X value
    training_sequences.append(text[i: i + sequence_length])
    # Grab the 41st character as the Y value to predict
    training_sequences_next_character.append(text[i + sequence_length])


# Convert letters to numbers to make training more efficient
X = np.zeros((len(training_sequences), sequence_length, len(chars)), dtype=np.bool)
y = np.zeros((len(training_sequences), len(chars)), dtype=np.bool)

for i, sentence in enumerate(training_sequences):
    for t, char in enumerate(sentence):
        X[i, t, chars_to_numbers[char]] = 1
    y[i, chars_to_numbers[training_sequences_next_character[i]]] = 1


# This function will sample the model and generate new text
def generate_new_text(epoch, _):
    seed_text = "when you have eliminated the impossible,"
    new_text = ""

    # Generate 1000 characters of new text
    for i in range(1000):

        # Encode the seed text as an array of numbers the same way our training data is encoded
        x_pred = np.zeros((1, sequence_length, len(chars)))
        for t, char in enumerate(seed_text):
            x_pred[0, t, chars_to_numbers[char]] = 1.

        # Predict which letter is most likely to come next
        predicted_letter_prob = model.predict(x_pred, verbose=0)[0]

        # Uncomment these lines to control the amount of randomness.
        # # Lower values make the model less random
        # randomness = 0.6
        # scaled_probabilities = np.exp(np.log(predicted_letter_prob) / randomness)
        # predicted_letter_prob = scaled_probabilities / np.sum(scaled_probabilities)

        # Hack to prevent sum of predictions from adding up to over 1.0 due to floating point precision issues
        predicted_letter_prob *= 0.99

        # Using the letter probabilities as weights, choose the next letter randomly
        next_index = np.argmax(np.random.multinomial(1, predicted_letter_prob, 1))

        # Look up the letter itself from it's index number
        next_char = numbers_to_chars[next_index]

        # Add the new letter to our new text.
        new_text += next_char

        # Update the seed text by dropping the first letter and adding the new letter.
        # This is so we can predict the next letter in the sequence.
        seed_text = seed_text[1:] + next_char

    # Print the new text we generated
    print(new_text)


# Set up the model
model = Sequential()
model.add(LSTM(128, input_shape=(sequence_length, len(chars)), return_sequences=True))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(len(chars), activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer="adam"
)

# Train the model
model.fit(
    X,
    y,
    batch_size=128,
    epochs=150,
    verbose=2,
    callbacks=[LambdaCallback(on_epoch_end=generate_new_text)]
)