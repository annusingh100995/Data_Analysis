import numpy as np 
import theano

import keras 
from matplotlib import pyplot as plt  

np.random.seed(123)

# Sequential is a stack of neural network layers
from keras.models import Sequential

# these are the different layer in keras and widely used in neural networks
from keras.layers import Dense, Dropout, Activation, Flatten

# these are the convloution layers
from keras.layers import Convolution2D, MaxPooling2D

# importing utilities
from keras.utils import np_utils 

# importing the dataset
from keras.datasets import mnist 

# Loading the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

""" Data is image data. Each image is 28 * 28 pixel wide
    Training set has 60000 samples 
    and test set has 10000 samples """


# we need to reshape the data , when using theano backend

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

# preprocessing the training data: includes setting the data type and normalising the values

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

y_train.shape

""" When we checked the shape of y_train is was a 1-d array.
    we need it to be into 10 different categories for 10 difeerent numbers
    basically one-hot encoded coding is required here. 
    we used np_utils to convert the data into differnt classes"""

Y_train = np_utils.to_categorical(y_train,10)
Y_test = np_utils.to_categorical(y_test,10)

model = Sequential()
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1,28,28)))
model.add((Convolution2D(32,(3,3), activation='relu', input_shape=(1,28,28))))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
    optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=3)