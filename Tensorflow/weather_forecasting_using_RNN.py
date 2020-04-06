# The recurrent models are computationly very expensive. 
# My pc cannot handel it. And it's lockdown period, I will not make my PC
# overwork. 

import os
import numpy as np 
import matplotlib.pyplot as plt


data_dir = r'D:\C++\PYTHON\ml\jena_climate'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

f = open(fname, encoding='utf8')
data = f.read()
f.close()

"""
Date Time","p (mbar)","T (degC)","Tpot (K)","Tdew (degC)","rh (%)","VPmax (mbar)","VPact (mbar)","VPdef (mbar)","sh (g/kg)","H2OC (mmol/mol)","rho (g/m**3)","wv (m/s)","max. wv (m/s)","wd (deg)"\n01.01.2009 00:10:00,996.52,-8.02,265.40,-8.90,93.30,3.33,3.11,0.22,1.94,3.12,1307.75,1.03,1.75,152.30\n01.01.2009 00:20:00,996.57,-8.41,265.01,-9.28,93.40,3.23,3.02,0.21,1.89,3.03,1309.80,0.72,1.50,136.10\n01.01.2009 00:30:00,996.53,-8.51,264.91,-9.31,93.90,3.21,3.01,0.20,1.88,3.02,1310.24,0.19,0.63,171.60\n01.01.2009 00:40:00,996.51,-8.31,265.12,-9.07,9
4.20,3.26,3.07,0.19,1.92,3.08,1309.19,0.34,0.50,198.00\n01.01.2009 00:50:00,996.51,-8.27,265.15,-9.04,94.10,3.27,3.08,0.19,1.92,3.09,1309.00,0.32,0.63,214.30\n01.01.2009 01:00:00,
996.50,-8.05,265.38,-8.78,94.40,3.33,3.14,0.19,1.96,3.15,1307.86,0.21,0.63,192.70\n01.01.2009 01:10:00,996.50,-7.62,
265.81,-8.30,94.80,3.44,3.26,0.18,2.04,3.27,1305.68,0.18,0.63,166.50\n01.01.2009 01:20:00,996.50,-7.62,265.81,-8.36,94.40,3.44,3.25,0.19,2.03,3.26,1305.69,0.19,'
"""
lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

print(header)
print(len(lines))
# 420551 lines, there are these many entries

# COnverting the data into Numpy arrays


float_data = np.zeros((len(lines), len(header)-1))
for i , line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i,:] = values

temp = float_data[:, 1]
plt.plot(temp)

# NOTICE THAT FLOAT DATA DOES NOT AHVE TIME 
# Preparing the data 
"""
The exact formulation of the problem will be as follows: given data going as far back
as lookback timesteps (a timestep is 10 minutes) and sampled every steps timesteps,
can you predict the temperature in delay timesteps? You’ll use the following parameter values:
 lookback = 720—Observations will go back 5 days.
 steps = 6—Observations will be sampled at one data point per hour.
 delay = 144—Targets will be 24 hours in the future.

Write a Python generator that takes the current array of float data and yields
batches of data from the recent past, along with a target temperature in the
future. Because the samples in the dataset are highly redundant (sample N and
sample N + 1 will have most of their timesteps in common), it would be wasteful
to explicitly allocate every sample.
"""

# use the first 200,000 timesteps as training data,
# so compute the mean and standard deviation only on this fraction of the data.

# Normalising the data
mean = float_data[:200000].mean(axis=0)
float_data -= mean 
std = float_data[:200000].std(axis=0)
float_data /= std


"""
The data generator  yields a tuple (samples, targets),
where samples is one batch of input data and targets is the corresponding array of
target temperatures. It takes the following arguments:

 data—The original array of floating-point data, that was normalized 
 lookback—How many timesteps back the input data should go.
 delay—How many timesteps in the future the target should be.
 min_index and max_index—Indices in the data array that delimit which timesteps to draw from. 
This is useful for keeping a segment of the data for validation and another for testing.
 shuffle—Whether to shuffle the samples or draw them in chronological order.
 batch_size—The number of samples per batch.
 step—The period, in timesteps, at which you sample data.  set it to 6 in
order to draw one data point every hour. Because data is redundant.

max_index = the furthers data index we can take, if we want to predit for 5 days, we have to remove the 5 days data and use 
the left over data for training


"""
"""
The thing is , we are basically adjusting the indices. We want to have enough data to move lookup steps backwards
and also enough data so that we can keep delay steps for testing. 
 
"""

def generator(data, lookback, delay, min_index, max_index,
    shuffle=False, batch_size=128, step = 6):
    if max_index is None:
        max_index = len(data)-delay-1
    i = min_index + lookback
    # i : we need to have minimum index from where we can start looking for data, because we are looking back for data
    # so we have have atleast lookup number of timesteps to create batches
    while 1:
        if shuffle:
            rows = np.random.randint(min_index+lookback, max_index, size = batch_size)
        else:
            if i+batch_size >= max_index:
                i = min_index+lookback
            rows = np.arange(i, min(i+batch_size,max_index))
            i += len(rows)
            # so i is array of index, rows + i
        samples = np.zeros((len(rows), lookback//step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j]- lookback, rows[j], step)
            samples[j] = data[indices]
            # since float_data does not ahve time, first column will give the temp.
            targets[j] = data[rows[j]+delay][1]
            #the target will be the future tempertaure , hence the index are larger then the samples
        yield samples, targets

lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = generator(float_data,lookback=lookback,delay=delay,
min_index=0,max_index=200000,shuffle=True,step=step,batch_size=batch_size)

val_gen = generator(float_data,lookback=lookback,delay=delay,min_index=200001,
max_index=300000,step=step,batch_size=batch_size)

test_gen = generator(float_data,lookback=lookback,delay=delay,min_index=300001,
max_index=None,step=step,batch_size=batch_size)

val_steps = (300000 - 200001 - lookback)
test_steps = (len(float_data) - 300001 - lookback)

# Tainign with densely connected model

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
model = Sequential()
model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,steps_per_epoch=50,epochs=10,
validation_data=val_gen,validation_steps=val_steps)

import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# Taking too much time

# GRU based model
# GRU are conceptionally same as LSTM only computationally cheaper 

from keras.optimizers import RMSprop
model = Sequential()
model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen, steps_per_epoch=50, epochs=10,
validation_data=val_gen , validation_steps=val_steps)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()



# Tackiling overfitting 

# Using dropout 
"""
 dropout,
which randomly zeros out input units of a layer in order to break happenstance (coincident)
correlations in the training data that the layer is exposed to. 

Every recurrent layer in Keras has two dropout-related
arguments: dropout, a float specifying the dropout rate for input units of the layer,
and recurrent_dropout, specifying the dropout rate of the recurrent units. 

"""
model = Sequential()

model.add(layers.GRU(32,dropout=0.2, recurrent_dropout=0.2,
input_shape=(None, float_data.shape[-1])))

model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loass='mae')
history = model.fit_generator(train_gen, steps_per_epoch=500,
epochs=40, validation_data=val_gen, validation_steps=val_steps)


"""
Now there is no overfitting, the step is to increase the capacity of the network. 

ncreasing network capacity is typically done by increasing the number of units in
the layers or adding more layers. Recurrent layer stacking is a classic way to build
more-powerful recurrent networks: 

To stack recurrent layers on top of each other in Keras, all intermediate layers
should return their full sequence of outputs (a 3D tensor) rather than their output at
the last timestep. This is done by specifying return_sequences=True

"""

# Dropiut regularised stacked GRU model

model = Sequential()
model.add(layers.GRS(32, dropout=0.1, recurrent_dropout=0.5,
return_sequences=True, input_shape=(None. float_data.shape[-1])))
model.add(layers.GRU(64, activation='relu', dropout=0.1,recurrent_dropout=0.5))
model.add(layers.Dense(1))
model.compile(optimizer='RMSprop()', loss='mae')
history = model.fit_generator(train_gen, steps_per_epoch=500, epcohs=40,
validation_data=val_gen, validation_steps=val_steps)

# Bidirectional RNNs
"""
A bidirectional RNN exploits the order sensitivity of RNNs: it consists of using two regular RNNs, such as the GRU and LSTM layers
,each of which processes the input sequence in one direction (chronologically and antichronologically), and then merging their representations. By processing a sequence both ways, a bidirectional RNN can catch patterns that
may be overlooked by a unidirectional RNN.
"""
# Training and evaluating an LSTM using reversed sequences

from keras.datasets import imdb
from keras.preprocessing import sequence
max_freature = 1000 
maxlen = 500
(x_train, y_train),(x_test, y_test) = imdb.load_data(num_words=max_freature)


x_train = [x[::-1] for x in x_train]
x_test  = [x[::-1] for x in x_test]

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

model = Sequential()
model.add(layers.Embedding(max_freature, 128))
model.add(layers.LSTM(32))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# the above traiing is just to show that the text processing in reveres 
# order works equally well as the original order

# Training and evaluating a bidirectional LSTM

model = Sequential()
model.add(layers.Embedding(max_freature, 32))
model.add(layers.Bidirectional(layers.LSTM(32)))
model.add(layers.Dense(1, activation = 'sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics =['acc'])
model.summary()
history = model.fit(x_train, y_train,epochs=10, batch_size=128, validation_split=0.2)

####
# Training a bidirectional GRU using the temperature data 


model = Sequential()
model.add(layers.Bidirectional(layers.GRU(32), input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))
model.compile(optimizer='RMSprop()', loss='mae')
history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=40, validation_data=val_gen,
validation_steps=val_steps)






