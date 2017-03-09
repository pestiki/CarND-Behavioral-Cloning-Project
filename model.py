import os
import csv

import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D

# Reading the csvfile for image entries

samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Creating validation set from training - Validation set size = 20% of training set
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# Defining a generator to reduce amount of GPU memory used.
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            correction_factor = 0.2  #  Correction factor for left and right steering angles.

            for batch_sample in batch_samples:
                for i in range(3):
                    name = './IMG/' + batch_sample[i].split('/')[-1]  # Reading the center, left and right
                                                                           # camera images
                    image = cv2.imread(name)
                    images.append(image)

                center_angle = float(batch_sample[3])
                angles.append(center_angle)
                angle_left = center_angle + correction_factor  # Correction  for left image steering angle
                angles.append(angle_left)
                angle_right = center_angle - correction_factor  # Correction for right image steering angle
                angles.append(angle_right)

            augmented_X, augmented_y = [], []

            for image, angle in zip(images, angles):  # Data augmentation on captured images of training set
                augmented_X.append(image)
                augmented_X.append((cv2.flip(image, 1)))    # Flipping images for augmenting pictures.
                augmented_y.append(angle)
                augmented_y.append(angle * -1.0)

            # trim image to only see section with road
            X_train = np.array(augmented_X)  # Creating Numpy arrays for features and labels
            y_train = np.array(augmented_y)
            yield shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


model = Sequential()
# Data normalization, centered around zero with small standard deviation
model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(160, 320, 3), output_shape=(160, 320, 3)))
# Cropping images to eliminated unnecessary data from features and defining the training network
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')  # Using Adam optimizer and Mean Square Error
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,
                    nb_val_samples=len(validation_samples), nb_epoch=3)  # training
model.save('model.h5')  # Saving model output in an h5 file
exit()
