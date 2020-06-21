#Importing required python libraries
import os
import librosa
import tensorflow as tf
from keras.models import load_model
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
#from playsound import playsound

model = load_model('my_model.h5')

label = ["car_horn","dog_bark","engine_idling","siren"]

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

#Extracting the feataures of a wav file as input to the data
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

test_sound = decodeFolder("test")
#test_sounds = np.concatenate(test_sound)
#print (test_sound.shape)

X_test = test_sound.reshape(test_sound.shape[0], test_sound.shape[1]).astype('float32')
#print (X_test)
#print (X_train.shape)

#for test in X_test:
listOfFiles = os.listdir("test")
#print (listOfFiles)

pred = model.predict_classes(X_test)
#print (pred)


for i in range (0, len(listOfFiles)):
	print ("Listening to",listOfFiles[i] )
	#playsound(("test\\"+str(listOfFiles[i])))
	print ("I think it is", label[pred[i]],"sound")
