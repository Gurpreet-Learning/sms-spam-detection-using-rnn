
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, cross_validation
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Embedding, LSTM,Dropout, BatchNormalization
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from keras.models import model_from_json
from keras.models import Sequential
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import pandas as pd
import random

class modeler:

	def __init__(self):
		self.dataset_main = self.load_dataset()
		self.max_features = 3000

	def load_dataset(self):
		
		data = pd.read_csv('files/spam_dataset.csv')
		data = data[['category','sentence']]
		return data


	def train_data(self):
		
		tokenizer = Tokenizer(nb_words=self.max_features, split=' ')
		tokenizer.fit_on_texts(self.dataset_main['sentence'].values)
		X = tokenizer.texts_to_sequences(self.dataset_main['sentence'].values)
		X = pad_sequences(X)
		Y = pd.get_dummies(self.dataset_main['category']).values
		X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X,Y, test_size = 0.33, random_state = 10)
		return X_train, X_test, Y_train, Y_test,X,Y


	def build_model(self,X,Y):
		embed_dim = 128
		lstm_out = 198
		model = Sequential()
		model.add(Embedding(self.max_features, embed_dim,input_length = X.shape[1], dropout=0.2))
		model.add(LSTM(lstm_out, dropout_U=0.5, dropout_W=0.5 ,return_sequences=True))
		model.add(LSTM(lstm_out, dropout_U=0.5, dropout_W=0.5 ,return_sequences=False))
		model.add(BatchNormalization())
		model.add(Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01)))
		model.add(Dropout(0.2))
		model.add(Dense(2,activation='softmax'))
		model.compile(loss = 'categorical_crossentropy', optimizer=Adam(lr=0.01),metrics = ['accuracy'])
		print(model.summary())
		return model

	def start_train(self):

		batch_size = 32
		X_train, X_test, Y_train, Y_test, X, Y = self.train_data()
		model = self.build_model(X,Y)
		early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')
		model.fit(X_train, Y_train, nb_epoch = 1, batch_size=batch_size, verbose = 1)
		self.save(model)

	def save(self,model):
		model_json = model.to_json()
		with open("Models/Model-Final.json", "w") as json_file:
			json_file.write(model_json)
		model.save_weights("Models/Model-Final.h5")
		print("Saved model to disk")



model = modeler()
model.start_train()
