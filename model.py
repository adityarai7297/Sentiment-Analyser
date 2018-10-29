from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import pickle
import pandas as pd


with open('xtrain.pkl', 'rb') as f:
   Xtrain = pickle.load(f)

with open('xtest.pkl', 'rb') as f1:
   Xtest = pickle.load(f1)

train = pd.read_csv("labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting=3)
Ytrain = train['sentiment'].tolist()

import numpy as np

Ytrain = np.asarray(Ytrain)


n_words = Xtest.shape[1]

model = Sequential()
model.add(Dense(50, input_shape=(n_words,), activation='relu'))
model.add(Dense(50, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(Xtrain, Ytrain, epochs=50, verbose=2)

from keras.models import load_model

model.save('trained_model.h5')  

#loss, acc = model.evaluate(Xtest, ytest, verbose=0)
#print('Test Accuracy: %f' % (acc*100))
