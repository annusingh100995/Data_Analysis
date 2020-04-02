from keras.datasets import reuters
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

""" 
 the Reuters dataset, a set of short newswires and their topics, published
by Reuters in 1986. It’s a simple, widely used toy dataset for text classification. There
are 46 different topics; some topics are more represented than others, but each topic
has at least 10 examples in the training set.
"""


word_index = reuters.get_word_index()
reversed_word_index = dict([(value , key) for (key, value) in word_index.items()])
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

decoded_newswire = ' '.join([reversed_word_index.get(i-3 , '?') for i in train_data[0]])

# Preparing the data

# Vectorising the data
import numpy as np 

def vectorize_sequences(sequences , dimension=10000): 
    results = np.zeros((len(sequences), dimension))
    for i , sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# One hot encoding the labels

def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), 46))
    for i , label in enumerate(labels):
        results[i,label] = 1
    return results

one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

#Inbuilt way of doing this

from keras.utils.np_utils import to_categorical
one_hot_test_labels = to_categorical(test_labels)
one_hot_train_labels = to_categorical(train_labels)

"""
    BUILDING THE NETWORK

    Since it is a multiclass classification the output classes are 46 in this case.

In a stack of Dense layers like that you’ve been using, each layer can only access information present 
in the output of the previous layer. If one layer drops some information 
relevant to the classification problem, this information can never be recovered by later
layers: each layer can potentially become an information bottleneck. In the previous
example, you used 16-dimensional intermediate layers, but a 16-dimensional space may
be too limited to learn to separate 46 different classes: such small layers may act as information bottlenecks, 
permanently dropping relevant information.
Hence larger layers are used.
"""

from keras import models 
from keras import layers 

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

"""
The last layer uses a softmax activation. You saw this pattern in the MNIST
example. It means the network will output a probability distribution over the 46
different output classes—for every input sample, the network will produce a 46-
dimensional output vector, where output[i] is the probability that the sample
belongs to class i. The 46 scores will sum to 1.
"""

model.compile(optimizer='rmsprop', loss = 'categorical_crossentropy',
    metrics=['accuracy'])

# Creating validation data

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

# Trainin the data for 20 epochs

history = model.fit(partial_x_train, partial_y_train,
    epochs = 20, batch_size = 512 ,validation_data=(x_val, y_val))


# use the built in encoder, the self coded isn't working fine i think 
# Plotting the taining and alidation loss

import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, 21)

plt.plot(epochs, loss, 'bo' , label ='Training loss')
plt.plot(epochs, val_loss, 'b' , label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc , 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b' , label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

results = model.evaluate(x_test, one_hot_test_labels)

