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
import tensorflow.keras.mixed_precision
import pandas as pd
from tensorflow.python.keras.layers import Activation

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


def predict(img_path, model_1, test_data, train_data, name):
    # Create the image
    print("Predicting for ", name, "...")
    for i in range(21):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("[%-20s] %d%%" % ('=' * i, 5 * i))
        sys.stdout.flush()
        sleep(0.1)
    print()

    # imag = image.load_img(img_path, target_size=(224, 224))
    # img_array = image.img_to_array(imag)
    # img_batch = np.expand_dims(img_array, 0)
    # prediction = model_1.predict(img_batch)
    # prediction2 = model_2.predict(img_batch)
    # prediction = (np.array(prediction).transpose()).dot(np.array(prediction2))
    # prediction = np.sum(prediction, 0).reshape((18, 1))
    # print(prediction)

    #
    # Prints
    #
    imag = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(imag)
    img_batch = np.expand_dims(img_array, 0)

    val_accuracy1 = history_1.history['val_accuracy']
    val_accuracy2 = history_2.history['val_accuracy']
    val_accuracy3 = history_3.history['val_accuracy']

    prediction = model_1.predict(img_batch)
    prediction *= val_accuracy1
    prediction2 = (model_2.predict(img_batch) * val_accuracy2)
    prediction3 = model_3.predict(img_batch) * val_accuracy3
    prediction += (prediction2 + prediction3)
    #
    # Prints
    #
    print("Predicting for ", name, " ...")
    sleep(2)
    predictedFood = np.array(np.argsort(prediction)[0])
    predictedFood = predictedFood.tolist()

    print("\nPrediction array: ", predictedFood)
    for x in range(3):
        for z in train_data.class_indices:
            if train_data.class_indices.get(z) == predictedFood[(len(predictedFood) - 1)]:
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

trainpath = "/Users/stuar/Desktop/TrainingData/FOOD101/images/train/"
testpath = "/Users/stuar/Desktop/TrainingData/FOOD101/images/test/"

data_dir = pathlib.Path(trainpath)
class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
print(class_names)
train_data = train_datagen.flow_from_directory(trainpath, batch_size=30, target_size=(224, 224), shuffle=True,
                                               class_mode="categorical")
test_data = valid_datagen.flow_from_directory(testpath, batch_size=30, target_size=(224, 224), shuffle=True,
                                              class_mode="categorical")

model_1 = tf.keras.models.Sequential([
    # SEE ARTICLE WE PRINTED
    layer.Conv2D(40, (5, 5), strides=(1, 1),
                           input_shape=(224, 224, 3), activation='relu'),
    # tf.keras.layers.Conv2D(30, (3, 3), activation='relu'),
    layer.MaxPooling2D((5, 5)),
    layer.Conv2D(40, (5, 5), activation='relu'),
    layer.MaxPooling2D((5, 5)),
    layer.Flatten(),
    layer.Dense(13, activation="softmax")
])
model_2 = tf.keras.models.Sequential([
    layer.Conv2D(30, (5, 5), strides=(1, 1),
                           input_shape=(224, 224, 3), activation='relu'),
    layer.MaxPooling2D((5, 5)),
    layer.Conv2D(30, (5, 5), activation='relu'),
    layer.MaxPooling2D((5, 5)),
    layer.Conv2D(30, (5, 5), activation='relu'),
    layer.Flatten(),
    layer.Dense(13, activation="softmax")
])


model_3 = tf.keras.models.Sequential([
    # SEE ARTICLE WE PRINTED
    layer.Conv2D(100, (3, 3), strides=(1, 1),
                           input_shape=(224, 224, 3), activation='relu'),
    # tf.keras.layers.Conv2D(30, (3, 3), activation='relu'),
    layer.MaxPooling2D((3, 3)),
    layer.Flatten(),
    layer.Dense(13, activation="softmax")
])

model_1.compile(tf.keras.optimizers.Adam(learning_rate=.001),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])
# model_1.trainable = False

history_1 = model_1.fit(train_data, epochs=3, steps_per_epoch=len(train_data),
                        validation_data=test_data, validation_steps=len(test_data))

model_2.compile(tf.keras.optimizers.Adam(learning_rate=.001),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])
history_2 = model_2.fit(train_data, epochs=3, steps_per_epoch=len(train_data),
                        validation_data=test_data, validation_steps=len(test_data))

model_3.compile(tf.keras.optimizers.Adam(learning_rate=.001),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])
history_3 = model_3.fit(train_data, epochs=3, steps_per_epoch=len(train_data),
                        validation_data=test_data, validation_steps=len(test_data))


plot_loss_curves(history_1)
# #
# Test region
#
pancake = "../Udemy/pancakes.jpg"
fries = "../Udemy/fries.jpg"
apple_pie = "../Udemy/apple_pie_test.jpg"
wings = "../Udemy/wings.jpg"
ribs = "../Udemy/ribs.jpg"

# Assign the path
img_path = pancake
img = Image.open(img_path)
plt.imshow(img)
model_1.save('../Udemy/models/foodCNN')
print("Classes: ", test_data.class_indices, "\n")
predict(pancake, model_1, test_data, train_data, "pancake")
predict(fries, model_1, test_data, train_data, "fries")
predict(apple_pie, model_1, test_data, train_data, "apple pie")
predict(wings, model_1, test_data, train_data, "wings")
predict(ribs, model_1, test_data, train_data, "ribs")
