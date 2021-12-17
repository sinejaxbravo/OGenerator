import datetime
import pathlib
import sys
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers import Activation

pancake = "../Udemy/pancakes.jpg"
fries = "../Udemy/fries.jpg"
apple_pie = "../Udemy/apple_pie_test.jpg"
wings = "../Udemy/wings.jpg"
ribs = "../Udemy/ribs.jpg"
reg_net = "https://tfhub.dev/adityakane2001/regnety600mf_classification/1"
efficient_net = "https://tfhub.dev/tensorflow/efficientnet/b7/classification/1"
res_net = "https://tfhub.dev/google/imagenet/resnet_v2_152/classification/5"

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

    imag = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(imag)
    img_batch = np.expand_dims(img_array, 0)
    prediction = model_1.predict(img_batch)
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


def create_tensor_callback(dir_name, experiment_name):
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"Saving files to {log_dir}!")
    return tensorboard_callback


# tf.random.set_seed(42)
train_datagen = ImageDataGenerator(rescale=1 / 255.,
                                   rotation_range=0.3,
                                   shear_range=0.1,
                                   zoom_range=0.3,
                                   width_shift_range=0.3,
                                   height_shift_range=0.3,
                                   horizontal_flip=True)

valid_datagen = ImageDataGenerator(rescale=1 / 255.)

trainpath = "../Udemy/data/TRANSFERFOOD101/train1/"
testpath = "../Udemy/data/TRANSFERFOOD101/test1/"

data_dir = pathlib.Path(trainpath)
class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
print(class_names)
train_data = train_datagen.flow_from_directory(trainpath, batch_size=32, target_size=(224, 224), shuffle=True,
                                               class_mode="categorical")
test_data = valid_datagen.flow_from_directory(testpath, batch_size=32, target_size=(224, 224), shuffle=True,
                                              class_mode="categorical")

#
#
# Create the model
#
#

def create_model(model_url, num_classes=18):
    return tf.keras.model.Sequential([
        hub.keras_layer(model_url, trainable=False, name="feature_extraction_layer", input_shape=(224, 224, 3)),
        tf.keras.layers.Dense(18, activation='softmax', name="output_layer")
    ])

model_1 = tf.keras.models.Sequential([
    # tf.keras.layers.Dense(18, activation=tf.nn.softmax),
    tf.keras.layers.Conv2D(60, 3,
                           input_shape=(224, 224, 3), activation='relu'),
    # tf.keras.layers.Conv2D(30, (3, 3), activation='relu'),

    tf.keras.layers.Conv2D(60, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(18, activation="softmax")
])
# model_2 = create_model(res_net, 18)
# model_3 = create_model(efficient_net, 18)
# model_4 = create_model(reg_net, 18)

#
#
# compile the model
#
#

model_1.compile(tf.keras.optimizers.Adam(learning_rate=.0001),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])

#
#
# fit the model
#
#

history_1 = model_1.fit(train_data, epochs=0, steps_per_epoch=len(train_data),
                        validation_data=test_data, validation_steps=len(test_data))


#
#
# post train actions
#
#

plot_loss_curves(history_1)
# #
# Test region
#


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
