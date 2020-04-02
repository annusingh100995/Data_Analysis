import tensorflow as tf
import numpy as np
import os
import time

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# READING THE DATA 
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

# Priniting the lenght of the text
print('Length of text : {} characters'. format(len(text)))

# sorting the unique characters of the text
vocab = sorted(set(text))
print('{} unique characters '. format(len(vocab)))


# vectorising the text 
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
 # the characters are converted into intergers
text_as_int = np.array([char2idx[c] for c in text])

print('{')
for char , _ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d}, '. format(repr(char), char2idx[char]))


print('.... \n')

# Showing the charters mapped to the integers 
print('{} --- charcter mapped to in ---> {}'. format(repr(text[:13]), text_as_int[:13]))

# THE PREDICTION TASK 
""" 
What is the most probable next character in a sequence, given a character or a sequence of charater. 
The model is being trained on this task.  

Input to the model : sequence of characters 
Output of the model : the next character at each time step 

RNN maintain an internal state that depends on the previously seen element. 
given all the characters computed until this moment. 

"""

# Creating trainig examples and targets

"""
Input sequence : contain the seq_length charaters from the text
Target sequence : conatin the same length of text except shifted one character to the rigth

sequence = hello 
Input sequence : hell
Target sequence : ello
"""

# the maximum length we want for a single input in characters
seq_length = 100 
examples_per_epoch = len(text)//(seq_length+1)

# creating training examples and targets 
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

for i in char_dataset.take(5):
    print(idx2char[i.numpy()])

"""
Batch method lets user to convert the individual characters to sequenceof desired size. 

"""

# text is divided into sequences for length sequence length
sequences = char_dataset.batch(seq_length+1, drop_remainder = True)

for item in sequences.take(5):
    print(repr(''. join(idx2char[item.numpy()])))

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

for input_example, target_example in dataset.take(1):
    print("Input data : ", repr(''.join(idx2char[input_example.numpy()])))
    print('Target data: ', repr(''.join(idx2char[target_example.numpy()])))

