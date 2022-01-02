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

dirtrain = "/Users/stuar/Desktop/TrainingData/squares/train"
dirtest = "/Users/stuar/Desktop/TrainingData/squares/test"
dir_large = "/Users/stuar/Desktop/TrainingData/unsupervised"
dir_one = "/Users/stuar/Desktop/TrainingData/unsup"



def residual(x, filters, k_size=1):
    xp = layer.ReLU()(x)
    xp = layer.BatchNormalization()(x)
    xp = layer.ReLU()(x)
    xp = layer.Conv2D(filters, (3, 3), activation='relu', padding='same')(xp)
    xp = layer.BatchNormalization()(xp)
    tot = layer.ReLU()(xp)
    xp = layer.Conv2D(filters, (3, 3), padding='same')(xp)
    tot = layer.Add()([xp, x])
    return tot


def block_norm(x, filters, kernel, strides):
    x = layer.ReLU()(x)
    m = layer.Conv2D(filters=filters, kernel_size=kernel, strides=strides)(x)
    norm = layer.BatchNormalization()(m)
    activation = layer.ReLU()(norm)
    return activation


# def block_pool(x, filters):
#     x =layer.ReLU()(x)
#     x =


def block_dense(x, output):
    x = layer.Dense(output * 2)(x)
    x = layer.Dense(output * 2)(x)
    x = layer.Dense(output)(x)
    return x


def sequence_a(x, filt, kernel, stride):
    # x = block_norm(x, filt, kernel, stride)
    x = residual(x, filt, kernel)
    return x


def concat(x, y):
    tot = layer.Add()(x)
    tot = layer.Add()([tot, y])
    tot = layer.ReLU()(tot)
    tot = layer.BatchNormalization()(tot)
    return tot


def get_directory(path, datagen):
    return datagen.flow_from_directory(path, batch_size=30, target_size=(224, 224), shuffle=True,
                                             class_mode="categorical")


def get_datagen(train_test):
    if train_test == "train":
        return ImageDataGenerator(rescale=1 / 255.,
                                  rotation_range=0.3,
                                  shear_range=0.1,
                                  zoom_range=0.3,
                                  width_shift_range=0.3,
                                  height_shift_range=0.3,
                                  horizontal_flip=True)
    else:
        return ImageDataGenerator(rescale=1 / 255.)


def predict(model, class_names):
    results = []
    stripes = []
    contrast = []
    solid = []
    graphics = []
    patterns = []

    for filename in os.listdir("clothes/pairs"):
        imag = image.load_img(f"clothes/pairs/{filename}", target_size=(224, 224))
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
    rmerge = []
    for r in results:
        r.sort()
        temp = []
        print("LEAST LIKELY: ")
        for z in range(10):
            print(f"{i}: {class_names[x]} {r.pop(0)}")
            i += 1
        r.sort(reverse=True)
        i = 1
        print("MOST LIKELY: ")
        for z in range(25):
            q = r.pop(0)
            print(f"{i}: {class_names[x]} {q}")
            temp.append(q[1])
            i += 1
        temp.sort()
        rmerge.append(temp)
        x += 1
        i = 1


    y = 0
    i = 0
    for r in rmerge:
        c = class_names[i]
        for x in r:
            for r1 in rmerge:
                c2 = class_names[y]
                if r != r1 and x in r1:
                    print(f"{x} in classes {c} and {c2}")
                y += 1
            y = 0
        i += 1




def build_CNN():
    train_datagen = get_datagen("train")

    valid_datagen = get_datagen("test")
    data_dir = pathlib.Path(dirtrain)
    class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
    print(class_names)

    train_data = get_directory(dirtrain, train_datagen)
    test_data = get_directory(dirtest, valid_datagen)

    input = layer.Input(shape=(224, 224, 3))

    x = layer.Conv2D(filters=256, kernel_size=7, strides=2, padding='same')(input)
    x = sequence_a(x, 256, 1, 1)
    x = layer.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = sequence_a(x, 128, 1, 1)
    x = layer.Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(x)
    x = sequence_a(x, 64, 1, 1)


    # x = sequence_a(x1, 16, 7, 2)
    # x = concat([block_norm(x, 16, 1, 1), block_norm(x, 16, 1, 1), block_norm(x, 16, 1, 1)], x)
    # x = sequence_a(x, 32, 3, 1)
    # x = concat([block_norm(x, 32, 1, 1), block_norm(x, 32, 1, 1), block_norm(x, 32, 1, 1)], x)

    # x = concat([block_norm(x, 64, 1, 1), block_norm(x, 64, 1, 1), block_norm(x, 64, 1, 1)], x)
    x = layer.Flatten()(x)
    x = layer.Dense(1024)(x)
    x = layer.Dense(1000)(x)

    output = layer.Dense(5, activation='softmax')(x)

    model = tf.keras.Model(inputs=input, outputs=output)

    model.compile(tf.keras.optimizers.Adam(learning_rate=.001),
                  loss=tf.keras.losses.CosineSimilarity(),
                  metrics=['accuracy'])

    history_1 = model.fit(train_data, epochs=7, steps_per_epoch=len(train_data),
                          validation_data=test_data, validation_steps=len(test_data))

    predict(model, class_names)


def oldCNN():
    data_dir = pathlib.Path(dirtrain)
    class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
    train_datagen = get_datagen("tin")
    valid_datagen = get_datagen("test")

    train_data = get_directory(dirtrain, train_datagen)
    test_data = get_directory(dirtest, valid_datagen)

    base = tf.keras.applications.ResNet152(input_shape=(224, 224, 3),
                                           include_top=False,
                                           weights='imagenet')
    base.trainable = False
    inputs = tf.keras.Input(shape=(224, 224, 3))
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    x1 = base(inputs, training=False)
    x = layer.Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(x1)
    x = sequence_a(x, 256, 1, 1)
    x = layer.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = sequence_a(x, 128, 1, 1)
    x = layer.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = sequence_a(x, 64, 1, 1)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.Dense(1000)(x)
    x = tf.keras.layers.Dense(5)(x)
    outputs = layer.Activation('softmax')(x)
    model = tf.keras.Model(inputs, outputs)

    print(len(train_data))
    model.compile(tf.keras.optimizers.Adam(learning_rate=.00001),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    history_1 = model.fit(train_data, epochs=10, steps_per_epoch=len(train_data),
                          validation_data=test_data, validation_steps=len(test_data))
    predict(model, class_names)
    return model




# build_CNN()
# oldCNN()