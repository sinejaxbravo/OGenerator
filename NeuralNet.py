import pathlib

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

# Plot the validation and training data separately
def plot_loss_curves(history):
    """
  Returns separate loss curves for training and validation metrics.
  """
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    # Plot loss
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()


def predict(image_prediction_path, model_1, prediction_data, name):
    # Create the image
    print("Predicting for ", name, "...")
    for i in range(21):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("[%-20s] %d%%" % ('=' * i, 5 * i))
        sys.stdout.flush()
        sleep(0.1)
    print()

    #
    # Prints
    #
    imag = image.load_img(image_prediction_path, target_size=(224, 224))
    img_array = image.img_to_array(imag)
    img_batch = np.expand_dims(img_array, 0)

    prediction = model.predict(img_batch)

    print("Predicting for ", name, " ...")
    sleep(2)
    predictedFood = np.array(np.argsort(prediction)[0])
    predictedFood = predictedFood.tolist()

    print("\nPrediction array: ", predictedFood)
    for x in range(3):
        for z in prediction_data.class_indices:
            if prediction_data.class_indices.get(z) == predictedFood[(len(predictedFood) - 1)]:
                if x == 0:
                    print("{}st most likely: ".format(x + 1), z)
                elif x == 1:
                    print("{}nd most likely: ".format(x + 1), z)
                else:
                    print("{}rd most likely: ".format(x + 1), z)
                predictedFood.pop()
                break
    print("DONE!\n")
    sleep(2)


#
#
# Start of code
#
#

# tf.random.set_seed(42)
train_datagen = ImageDataGenerator(rescale=1 / 255.,
                                   rotation_range=0.3,
                                   shear_range=0.1,
                                   zoom_range=0.3,
                                   width_shift_range=0.3,
                                   height_shift_range=0.3,
                                   horizontal_flip=True)

valid_datagen = ImageDataGenerator(rescale=1 / 255.)

dirtrain = "/Users/stuar/Desktop/TrainingData/FashionGen/train"
dirtest = "/Users/stuar/Desktop/TrainingData/FashionGen/test"

data_dir = pathlib.Path(dirtrain)
class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
print(class_names)
train_data = train_datagen.flow_from_directory(dirtrain, batch_size=30, target_size=(224, 224), shuffle=True,
                                               class_mode="categorical")
test_data = valid_datagen.flow_from_directory(dirtest, batch_size=30, target_size=(224, 224), shuffle=True,
                                              class_mode="categorical")

base = tf.keras.applications.ResNet152(input_shape=(224, 224, 3),
                                         include_top=False,
                                         weights='imagenet')
base.trainable = False
inputs = tf.keras.Input(shape=(224, 224, 3))
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

x = base(inputs, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
prediction_layer = tf.keras.layers.Dense(2)(x)
outputs = layer.Activation('softmax')(prediction_layer)
model = tf.keras.Model(inputs, outputs)

print(len(train_data))
model.compile(tf.keras.optimizers.Adam(learning_rate=.0001),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

history_1 = model.fit(train_data, epochs=5, steps_per_epoch=len(train_data),
                      validation_data=test_data, validation_steps=len(test_data))

plot_loss_curves(history_1)
# plot_loss_curves(history_2)
# plot_loss_curves(history_3)
# #
# Test region
#
imag = image.load_img("IMG_0655.jpg", target_size=(224, 224))
img_array = image.img_to_array(imag)
img_batch = np.expand_dims(img_array, 0)
predstyle = model.predict(img_batch)
imag = image.load_img("IMG_0656.jpg", target_size=(224, 224))
img_array = image.img_to_array(imag)
img_batch = np.expand_dims(img_array, 0)

predictionugly = model.predict(img_batch)

print(predstyle)
print(predictionugly)