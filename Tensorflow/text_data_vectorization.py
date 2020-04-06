import numpy as np 

# ONE HOT ENCODING FOR WORDS
samples = ['The cat sat on the mat.' , 'The dog ate my homework.']

token_index = {}

# Creating a dictionary for word and giving them an index
for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index)+1

max_length = 10 

# Creating the tensor 
# len(samples) the number of dcouments. here there are two sample in the text 
# max length = is the number of unique words, this is the lenght of the vector
# max(token_index) is a vector for each word 
 
results = np.zeros(shape=(len(samples), max_length, max(token_index.values())+1))

for i , sample in enumerate(samples):
    for j , word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i,j,index] = 1.


# ONE HOT ENCODING FOR CHARACTERS

import string

# max_length_char is the number of unique characters
max_length_char = 50

characters = string.printable
# Dictionary with numerical index for each character

token_index_char = dict(zip(range(1, len(characters)+1), characters))

results_char = np.zeros((len(samples), max_length_char, max(token_index_char.keys())+1))

for i , sample in enumerate(samples):
    for j , characters in enumerate(sample):
        indec = token_index_char.get(characters)
        results_char[i,j,index] = 1.

# one hot encoding using keras

from keras.preprocessing.text import Tokenizer

# Creates a tokenizer, configured to take into account the 1000 most comman words

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(samples)

# Convert strings into list of integer indices
sequences = tokenizer.texts_to_sequences(samples)
# [[1, 2, 3, 4, 1, 5], [1, 6, 7, 8, 9]]

one_hot_results = tokenizer.texts_to_matrix(samples , mode='binary')
""""
array([[0., 1., 1., ..., 0., 0., 0.],
       [0., 1., 0., ..., 0., 0., 0.]])
>>> one_hot_results .shape
(2, 1000)

"""

word_index = tokenizer.word_index
# {'the': 1, 'cat': 2, 'sat': 3, 'on': 4, 'mat': 5, 'dog': 6, 'ate': 7, 'my': 8, 'homework': 9}

# WORD EMBEDDINGS
"""
word embeddings are lowdimensional floating-point vectors.
word embeddings are learned from data. It’s common to see word embeddings that are
256-dimensional, 512-dimensional, or 1,024-dimensional when dealing with very large
vocabularies. On the other hand, one-hot encoding words generally leads to vectors
that are 20,000-dimensional or greater (capturing a vocabulary of 20,000 tokens, in
this case). So, word embeddings pack more information into far fewer dimensions.

There are two ways to obtain word embeddings:
 Learn word embeddings jointly with the main task you care about (such as document classification or sentiment prediction). In this setup, you start with random word vectors and then learn word vectors in the same way you learn the
weights of a neural network.
 Load into your model word embeddings that were precomputed using a different machine-learning task than the one you’re trying to solve. These are called
pretrained word embeddings.

"""

from keras.layers import Embedding

embedding_layer = Embedding(1000, 64)
"""
The Embedding layer takes at least two
arguments: the number of possible tokens
(here, 1,000: 1 + maximum word index)
and the dimensionality of the embeddings
(here, 64)

The Embedding layer is best understood as a dictionary that maps integer indices
(which stand for specific words) to dense vectors. It takes integers as input, it looks up
these integers in an internal dictionary, and it returns the associated vectors. It’s effectively a dictionary lookup

"""

from keras.datasets import imdb 
from keras import preprocessing

# the max number of words for the feature
max_features = 10000

# cut off text after this number of words (among the max feature most common words)
maxlen = 20

# Load the data as a list of integers
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# turns the list of integers into a 2D integar tensor if shape (samples, maxlen)
# means we are taking only the first maxlen number of words fron the actual review 
# for further modelling
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)

# Using an Embedding layer and classifier on the IMDB data

from  keras.models import Sequential
from keras.layers import Flatten, Dense



# Specifies the maximum input layer to the Embedding layer
# so that this layer can be flatten later
# After the embedding layer the activation have shape (samples, maxlen , 8)

model = Sequential()
model.add(Embedding(10000, 8, input_length=maxlen))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

model.summary()

history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

val_acc = history.history['val_acc']
acc = history.history['acc']

import matplotlib.pyplot as plt

epochs = range(1,11)
plt.plot(epochs, val_acc, 'bo', label='val_acc')
plt.plot(epochs, acc, 'b', label='acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Pretrained word embeddings

# Processing the raw IMDB data

import os 
imbd_dir = r'D:\C++\PYTHON\ml\aclImdb'
train_dir = os.path.join(imbd_dir, 'train')

labels = []
texts = []

# Loading the text and relabeliing neg to 0 and pos to 1

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname), encoding="utf8")
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

# Tokenizing the text 

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np 

# Cuts the reviews after 100 words
max_len = 100

# Taking inly 200 samples for train data , because pretrained word embeddings
# are used when there is less training data 
training_samples  = 200

# validation samples
validation_samples = 10000

# consider only the top 10000 unique words from the common words
max_words = 10000


tokenizer = Tokenizer(num_words= max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

data = pad_sequences(sequences, maxlen=max_len)

labels = np.asarray(labels)

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# randomly shuffling the data because first all reviews are 0 and then 1s

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = data[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]

x_val = data[training_samples: training_samples + validation_samples]

y_val = labels[training_samples: training_samples + validation_samples]

# Downloading the GloVe word embedding
glove_dir = r'D:\C++\PYTHON\ml\glove.6B'

embeddings_index = {}

f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding='utf8')
for line in f:
    values = line.split()
    words = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[words] = coefs


f.close()

print('Found %s word vectors.' %len(embeddings_index))

# Building the embedding matrix

"""
The embedding matrix is of the shape (max_words , embeddings_index)
where enach entry i contains the embedding_dim - dimensional vector
for the word of index i in the reference word index. 

So basically, embedding matris contains vectors of pre-defined length and 
dimensionality for each word in the word index prepare during tokenization 

"""

embedding_dim = 100 
embedding_matrix = np.zeros((max_words , embedding_dim))

for word , i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# for each word in word index, till i is less then the max words to be used for taining network, 10000 here
# find the word in embedding_index
# if that word is found store it in the embedding layer


# Defining a model

from keras.models import Sequential
from keras.layers import Embedding, Dense, Flatten

model = Sequential()
model.add(Embedding(max_words, embedding_dim , input_length=max_len))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

#
# The Embedding layer has a single weight matrix: a 2D float matrix where each entry i is
# the word vector meant to be associated with index i

# Loading pretrained word embedding into the embeddin layer

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

# freeze the Embedding layer (set its trainable attribute to False)
# the layer is pretrained hence it should not be trained during training

# Training the model

model.compile(optimizer='rmsprop' , loss='binary_crossentropy', metrics=['acc'])

history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))

model.save_weights('pre_trained_glove_model.h5')

# Plotting the results

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,11)

plt.plot(epochs, acc, 'bo' , label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label = 'Validation Acuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(epochs, loss, 'bo' , label='Training Loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Evaluating the model on test data

model.load_weights('pre_trained_glove_model.h5')
model.evaluate(x_test, y_test)

# Training without pretrained word embedding

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=max_len))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='rmsprop', loss= 'binary_crossentropy' , metrics=['acc'])
history = model.fit(x_train, y_train, epochs = 10, batch_size=32, validation_data=(x_val, y_val))

