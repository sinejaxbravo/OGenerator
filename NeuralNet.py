import os
import pathlib
from collections import Set
from queue import PriorityQueue

from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from time import sleep
import sys
import tensorflow.keras.layers as layer
from tensorflow.keras import mixed_precision
import pandas as pd
from tensorflow.python.keras.layers import Activation

# mixed_precision.set_global_policy("mixed_float16")
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


def residual(x, filters, k_size=1):
    xp = layer.Conv2D(filters, kernel_size=k_size, activation='relu', padding='same')(x)
    xp = layer.BatchNormalization()(xp)
    xp = layer.Conv2D(filters, kernel_size=k_size, padding='same')(xp)
    x = layer.Dropout(.2)(x)
    tot = layer.Add()([x, xp])
    tot = layer.ReLU()(tot)
    tot = layer.BatchNormalization()(tot)
    return tot


def block_norm(x, filters, kernel, strides):
    x = layer.ReLU()(x)
    m = layer.Conv2D(filters=filters, kernel_size=kernel, strides=strides)(x)
    norm = layer.BatchNormalization()(m)
    activation = layer.ReLU()(norm)
    return activation


def block_dense(x, output):
    x = layer.Dense(output * 2)(x)
    x = layer.Dense(output * 2)(x)
    x = layer.Dense(output)(x)
    return x


def sequence_a(x, filt, kernel, stride):
    x = block_norm(x, filt, kernel, stride)
    x = residual(x, filt, kernel)
    return x


def concat(x, y):
    tot = layer.Add()(x)
    tot = layer.Add()([tot, y])
    tot = layer.ReLU()(tot)
    tot = layer.BatchNormalization()(tot)
    return tot




# tf.random.set_seed(42)
train_datagen = ImageDataGenerator(rescale=1 / 255.,
                                   rotation_range=0.3,
                                   shear_range=0.1,
                                   zoom_range=0.3,
                                   width_shift_range=0.3,
                                   height_shift_range=0.3,
                                   horizontal_flip=True)

valid_datagen = ImageDataGenerator(rescale=1 / 255.)

dirtrain = "/Users/stuar/Desktop/TrainingData/squares/train"
dirtest = "/Users/stuar/Desktop/TrainingData/squares/test"

data_dir = pathlib.Path(dirtrain)
class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
print(class_names)
train_data = train_datagen.flow_from_directory(dirtrain, batch_size=30, target_size=(60, 60), shuffle=True,
                                               class_mode="categorical")
test_data = valid_datagen.flow_from_directory(dirtest, batch_size=30, target_size=(60, 60), shuffle=True,
                                              class_mode="categorical")

input = layer.Input(shape=(60, 60, 3))

x = layer.Conv2D(filters=32, kernel_size=3, strides=2, padding='same')(input)
x = layer.BatchNormalization()(x)
x1 = layer.ReLU()(x)

x = sequence_a(x1, 64, 7, 1)
x = concat([block_norm(x, 64, 1, 1), block_norm(x, 64, 1, 1), block_norm(x, 64, 1, 1)], x)
x = sequence_a(x, 256, 5, 5)
x = concat([block_norm(x, 256, 1, 1), block_norm(x, 256, 1, 1), block_norm(x, 256, 1, 1)], x)
x = sequence_a(x, 512, 3, 1)
x = concat([block_norm(x, 512, 1, 1), block_norm(x, 512, 1, 1), block_norm(x, 512, 1, 1)], x)
x = sequence_a(x, 1024, 1, 3)
x = layer.GlobalAveragePooling2D()(x)
x = layer.Dense(512)(x)
output = layer.Flatten()(x)
output = layer.Dense(5, activation='softmax')(output)

model = tf.keras.Model(inputs=input, outputs=output)

model.compile(tf.keras.optimizers.Adam(learning_rate=.001),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

history_1 = model.fit(train_data, epochs=20, steps_per_epoch=len(train_data),
                      validation_data=test_data, validation_steps=len(test_data))

results = []
stripes = []
contrast = []
solid = []
graphics = []
patterns = []


for filename in os.listdir("clothes/pairs"):
    imag = image.load_img(f"clothes/pairs/{filename}", target_size=(60, 60))
    img_array = image.img_to_array(imag)
    img_batch = np.expand_dims(img_array, 0)
    prediction = model.predict(img_batch)
    contrast.append((prediction[0][0], filename))
    graphics.append((prediction[0][1], filename))
    solid.append((prediction[0][2], filename))
    patterns.append((prediction[0][3], filename))
    stripes.append((prediction[0][4], filename))

results = [contrast, graphics, solid, patterns, stripes]
i = 1
x = 0
for r in results:
    r.sort()
    print("LEAST LIKELY: ")
    for z in range(10):
        print(f"{i}: {class_names[x]} {r.pop(0)}")
        i += 1
    r.sort(reverse=True)
    i = 1
    print("MOST LIKELY: ")
    for z in range(10):
        print(f"{i}: {class_names[x]} {r.pop(0)}")
        i += 1
    x += 1
    i = 1
