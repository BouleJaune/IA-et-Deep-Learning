import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
from tensorflow.examples.tutorials.mnist import input_data

from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.initializers import TruncatedNormal

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

img_size = 28
img_size_flat = 784
img_shape = [28,28]
img_shape_full = [28,28,1]
n_classes = 10
num_channels = 1


n_layers = 2
n_neurones = []
n_neurones.extend([484]*n_layers)
n_neurones.append(n_classes)

ini = TruncatedNormal(mean=0.0, stddev=0.1, seed=None)
optimizer = Adam(lr=1e-3)

model = Sequential()
model.add(InputLayer(input_shape=(img_size_flat,)))


for i in range(n_layers-1):    
    model.add(Dense(n_neurones[i],
                kernel_initializer=ini,
                bias_initializer=ini,activation='relu'))
                
model.add(Reshape([22,22,1]))                
model.add(Conv2D(kernel_size=5, strides=1, filters=16, padding='same',
                 activation='relu', name='layer_conv1'))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Conv2D(kernel_size=5, strides=1, filters=36, padding='same',
                 activation='relu', name='layer_conv2'))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))       
model.add(Dense(10, activation='softmax'))


model.compile(optimizer=optimizer,loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(mnist.train.images, mnist.train.labels, epochs = 10, batch_size=200)
result = model.evaluate(x=mnist.test.images,y=mnist.test.labels)

print("{0}: {1:.2%}".format(model.metrics_names[1], result[1]))

