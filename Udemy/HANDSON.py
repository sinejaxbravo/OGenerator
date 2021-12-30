import pathlib

import keras
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


def residual(x, filters, strides, k_size=1):
    xp = layer.Conv2D(filters, kernel_size=k_size, activation='relu', padding='same')(x)
    xp = layer.BatchNormalization()(xp)
    xp = layer.Conv2D(filters, kernel_size=k_size, padding='same')(xp)
    tot = layer.Add()([x, xp])
    tot = layer.ReLU()(tot)
    tot = layer.BatchNormalization()(tot)
    return tot


def block_norm(x, filters, strides, k_size=1):
    x = layer.ReLU()(x)
    m = layer.Conv2D(filters=filters, kernel_size=k_size, strides=strides)(x)
    norm = layer.BatchNormalization()(m)
    activation = layer.ReLU()(norm)
    return activation


def block_pool(x, filters, strides, k_size=1):
    x = layer.ReLU()(x)
    m = layer.Conv2D(filters=filters, kernel_size=k_size, strides=strides)(x)
    norm = layer.MaxPooling2D(pool_size=k_size)(m)
    activation = layer.ReLU()(norm)
    return activation

def block_dense(x, output):
    x = layer.Dense(output * 2)(x)
    x = layer.Dense(output * 2)(x)
    x = layer.Dense(output)(x)
    return x


def sequence_a(x, filt, stride, kernel):
    x = block_norm(x, filt, stride, kernel)
    x = residual(x, filt, stride, kernel)
    return x


def sequence_b(x, filt, stride, kernel):
    x = block_pool(x, filt, stride, kernel)
    x = residual(x, filt, stride, kernel)
    return x

# TODO make sure that the norm and pool align!!! Can also see if you can residual the whole concat
#  or "x = residual(concat)...?
def concat(x, filt, stride, kernel):
    s1 = block_pool(x, filt, stride, kernel)
    s1 = block_pool(s1, filt, stride, kernel)
    s1 = block_pool(s1, filt, stride, kernel)

    s2 = block_norm(x, filt, stride, kernel)
    s2 = block_norm(s2, filt, stride, kernel)
    s2 = block_norm(s2, filt, stride, kernel)

    tot = layer.Add()([s1, s2])
    tot = layer.ReLU()(tot)
    tot = layer.BatchNormalization()(tot)
    return tot


#
#
# Start of code
#
#

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

x = sequence_a(x1, 64, 1, 7)
x = sequence_a(x, 256, 5, 5)
x = sequence_a(x, 512, 1, 3)
x = sequence_a(x, 1024, 3, 1)
x = layer.Dense(1024)(x)

output = layer.Flatten()(x)
output = layer.Dense(5, activation='softmax')(output)

model = tf.keras.Model(inputs=input, outputs=output)

model.compile(tf.keras.optimizers.Adam(learning_rate=.001),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

history_1 = model.fit(train_data, epochs=12, steps_per_epoch=len(train_data),
                      validation_data=test_data, validation_steps=len(test_data))
print(history_1.history)

# f = open("model.txt", "a")
# f.write("\n\n\n\n\n")
# f.write(str(history_1.history))
# model.summary()
# f.close()

# plot_loss_curves(history_2)
# plot_loss_curves(history_3)
# #
# Test region
#
