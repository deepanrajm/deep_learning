#Importing required python libraries
import os
import librosa
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

#Defining One-Hot as labels
car_horn_onehot 		= [1,0,0,0]
dog_bark_onehot 		= [0,1,0,0]
engine_idling_onehot 	= [0,0,1,0]
siren_onehot 			= [0,0,0,1]


#Converting files in a folder into list of arrays containg the properties of the files
def decodeFolder(category):
	print("Starting decoding folder "+category+" ...")
	listOfFiles = os.listdir(category)
	arrays_sound = np.empty((0,193))
	for file in listOfFiles:
		filename = os.path.join(category,file)
		features_sound = extract_feature(filename)
		arrays_sound = np.vstack((arrays_sound,features_sound))
	return arrays_sound

#Extracting the feataures of a wav file as inpurt to the data
def extract_feature(file_name):
	print("Extracting "+file_name+" ...")
	X, sample_rate = librosa.load(file_name)
	stft = np.abs(librosa.stft(X))
	mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
	chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
	mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
	contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
	tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sample_rate).T,axis=0)
	return np.hstack((mfccs,chroma,mel,contrast,tonnetz))

#train data
car_horn_sounds = decodeFolder("car_horn")
car_horn_labels = [car_horn_onehot for items in car_horn_sounds]

dog_bark_sounds = decodeFolder("dog_bark")
dog_bark_labels = [dog_bark_onehot for items in dog_bark_sounds]

engine_idling_sounds = decodeFolder("engine_idling")
engine_idling_labels = [engine_idling_onehot for items in engine_idling_sounds]

siren_sounds = decodeFolder("siren")
siren_labels = [siren_onehot for items in siren_sounds]


train_sounds = np.concatenate((car_horn_sounds, dog_bark_sounds,engine_idling_sounds,siren_sounds))
train_labels = np.concatenate((car_horn_labels, dog_bark_labels,engine_idling_labels,siren_labels))
print (train_sounds.shape)
X_train = train_sounds.reshape(train_sounds.shape[0], train_sounds.shape[1]).astype('float32')
#X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
#test_data

test_sound = decodeFolder("test")
#test_sounds = np.concatenate(test_sound)
print (test_sound.shape)
X_test = test_sound.reshape(test_sound.shape[0], test_sound.shape[1]).astype('float32')
#print (X_train.shape)


model = Sequential()
model.add(Dense(193, input_dim=193, init='uniform', activation='relu'))
#model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(4, init='uniform', activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X_train, train_labels, nb_epoch=150, batch_size=10)

model.save('my_model.h5')

print (model.predict_classes(X_test))