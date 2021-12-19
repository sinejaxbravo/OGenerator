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


def identity_block(x, filter_size):
    x_skip = x
    x = layer.Conv2D(filter_size, (3, 3), padding='same')(x)
    x = layer.BatchNormalization(axis=3)(x)
    x = layer.Conv2D(filter_size, (3, 3), padding='same')(x)
    x = layer.BatchNormalization(axis=3)(x)
    x = layer.Conv2D(filter_size, (3, 3), padding='same')(x)
    x = layer.BatchNormalization()
    x = tf.keras.layers.Activation('relu')(x)
    return x


def convolutional_block(x, filter_size):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = tf.keras.layers.Conv2D(filter_size, (3, 3), padding='same', strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv2D(filter_size, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    # Processing Residue with conv(1,1)
    x_skip = tf.keras.layers.Conv2D(filter_size, (1, 1), strides=(2, 2))(x_skip)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)
    return x



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


shape = (224, 224, 3)

# Step 1 (Setup Input Layer)
x_input = tf.keras.layers.Input(shape)
x = tf.keras.layers.ZeroPadding2D((3, 3))(x_input)
# Step 2 (Initial Conv layer along with maxPool)
x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
# Define size of sub-blocks and initial filter size
block_layers = [3, 4, 6, 3]
filter_size = 64
# Step 3 Add the Resnet Blocks
for i in range(4):
    if i == 0:
        # For sub-block 1 Residual/Convolutional block not needed
        for j in range(block_layers[i]):
            x = identity_block(x, filter_size)
    else:
        # One Residual/Convolutional Block followed by Identity blocks
        # The filter size will go on increasing by a factor of 2
        filter_size = filter_size * 2
        x = convolutional_block(x, filter_size)
        for j in range(block_layers[i] - 1):
            x = identity_block(x, filter_size)
# Step 4 End Dense Network
x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dense(13, activation='softmax')(x)
model = tf.keras.models.Model(inputs=x_input, outputs=x, name="ResNet34")

model.compile(tf.keras.optimizers.Adam(learning_rate=.0001),
                   loss=tf.keras.losses.CategoricalCrossentropy(),
                   metrics=['accuracy'])

history_1 = model.fit(train_data, epochs=7, steps_per_epoch=len(train_data),
                           validation_data=test_data, validation_steps=len(test_data))





plot_loss_curves(history_1)
# plot_loss_curves(history_2)
# plot_loss_curves(history_3)
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
# model_1.save('../Udemy/models/foodCNN')
print("Classes: ", test_data.class_indices, "\n")
predict(pancake, model, train_data, "pancake")
predict(fries, model, train_data, "fries")
predict(apple_pie, model, train_data, "apple pie")
predict(wings, model, train_data, "wings")
predict(ribs, model, train_data, "ribs")
