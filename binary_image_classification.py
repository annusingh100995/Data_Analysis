import os, shutil

original_dataset_dir = 'D:\C++\PYTHON\ml\TF'
original_dataset_dir_cats = 'D:\C++\PYTHON\ml\TF\kagglecatsanddogs_3367a\PetImages\Cat'
original_dataset_dir_dogs = 'D:\C++\PYTHON\ml\TF\kagglecatsanddogs_3367a\PetImages\Dog'

base_dir = 'D:\C++\PYTHON\ml\TF\cats_and_dogs_small'
os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)

test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)


train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)



# Copy the first 1K cats image to train_cats_dir
fnames = ['{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_cats, fname)
    dst = os.path.join(train_cats_dir,fname)
    shutil.copyfile(src, dst)

# copying 500 images to validation_cats_dir
fnames = ['{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_cats, fname)
    dst = os.path.join(validation_cats_dir,fname)
    shutil.copyfile(src, dst)

# copying imgaes to test_cats_dir
fnames = ['{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_cats, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)

# same for dogs

fnames = ['{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_dogs, fname)
    dst = os.path.join(train_dogs_dir,fname)
    shutil.copyfile(src, dst)

# copying 500 images to validation_cats_dir
fnames = ['{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_dogs, fname)
    dst = os.path.join(validation_dogs_dir,fname)
    shutil.copyfile(src, dst)

# copying imgaes to test_cats_dir
fnames = ['{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_dogs, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)



print('Total Training Cat Images : ', len(os.listdir(train_cats_dir)))
print('Total Test Cat Images : ', len(os.listdir(test_cats_dir)))
print('Total Validation Cat Images : ', len(os.listdir(validation_cats_dir)))

print('Total Training Dog Images : ', len(os.listdir(train_dogs_dir)))
print('Total Test Dog Images : ', len(os.listdir(test_dogs_dir)))
print('Total Validation Dog Images : ', len(os.listdir(validation_dogs_dir)))

# All izz well

# BUILDING THE MODEL

from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()


from keras import optimizers
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

# Reading the images from directories

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen =ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
target_size=(150,150), batch_size=20, class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_dir,
target_size=(150,150), batch_size=20, class_mode='binary')

history = model.fit_generator(train_generator, steps_per_epoch=20,epochs=30, validation_data=validation_generator, validation_steps=30)

model.save(r'D:\C++\PYTHON\ml\TF\cats_dogs_small_1.h5')

# Displaying the accuracy and loss during training

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accurracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# The model suffers from overfitting 
# Augmentation is one way to tackle overfitting
"""
Data augmentation takes the approach of generating more training data
from existing training samples, by augmenting the samples via a number of random
transformations that yield believable-looking images. The goal is that at training time,
your model will never see the exact same picture twice. This helps expose the model
to more aspects of the data and generalize better

"""
# Setting up a data sugmentation configuration using ImageGenerator

datagen = ImageDataGenerator(rotation_range=40,
width_shift_range=0.2, height_shift_range=0.2,
shear_range=0.2, zoom_range=0.2,horizontal_flip=True,
fill_mode='nearest')

# Displaying randomly augmented training samples

# Module with image processing utilites
from keras.preprocessing import image


fname = [os.path.join(train_cats_dir, fname) 
for fname in os.listdir(train_cats_dir)]

# Choosing a random image to augment
img_path = fname[3]
# Read the image and resize it
img = image.load_img(img_path, target_size=(150,150))

# Converting the image to numpy array
x = image.img_to_array(img)

x = x.reshape((1,)+x.shape)

i = 0
# Generate batches of randomly transformed images.
# Loops indefinitiely , so breaking at a random point
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i%4 ==0:
        break

plt.show()

# Training the covnet using data - augmentation genration 

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
optimizer=optimizers.RMSprop(lr=1e-4),
metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=0.2,
height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,HOW DO YOU THINK MACHINE LEARNING COULD BE APPLIED TO A SCIENTIFIC DOMAIN E.G. BIOLOGY, PHYSICS, CHEMISTRY, MATHS (100-150 WORDS)horizontal_flip=True,)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,target_size=(150, 150),batch_size=32,
class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
validation_dir,target_size=(150, 150),batch_size=32,class_mode='binary')

history = model.fit_generator(train_generator,steps_per_epoch=20,epochs=20,
validation_data=validation_generator,validation_steps=20)

model.save('cats_and_dogs_small_2.h5')
