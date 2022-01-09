import pathlib
import os
import time

from six.moves import urllib
import tensorflow_datasets.public_api as tfds
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import tensorflow.keras.layers as layer

# mixed_precision.set_global_policy("mixed_float16")
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

dirtrain = "/Users/stuar/Desktop/TrainingData/dualclass/train"
dirtest = "/Users/stuar/Desktop/TrainingData/dualclass/test"
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


def incept(x, filters):
    # x = layer.Conv2D(filters=filters, kernel_size=7)(x)

    x1 = layer.Conv2D(filters=filters, kernel_size=1, strides=1, activation="relu", padding="same")(x)
    x1 = layer.Conv2D(filters=filters, kernel_size=3, strides=1, activation="relu", padding="same")(x1)
    x1 = layer.Conv2D(filters=filters, kernel_size=3, strides=1, activation="relu", padding="same")(x1)

    x2 = layer.Conv2D(filters=filters, kernel_size=1, strides=1, activation="relu", padding="same")(x)
    x2 = layer.Conv2D(filters=filters, kernel_size=3, strides=1, activation="relu", padding="same")(x2)

    x3 = layer.Conv2D(filters=filters, kernel_size=1, strides=1, activation="relu", padding="same")(x)

    x4 = layer.Conv2D(filters=filters, kernel_size=1, strides=1, activation="relu", padding="same")(x)

    x = layer.Add()([x1, x2, x3, x4])
    x = layer.AveragePooling2D(filters, padding="same")(x)
    return x






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
                                  height_shift_range=0.3)
    else:
        return ImageDataGenerator(rescale=1 / 255.)


def predict(model, class_names):
    results = []
    stripes = []
    contrast = []
    solid = []
    graphics = []
    patterns = []

    for filename in os.listdir("clothes/pair"):
        imag = image.load_img(f"clothes/pair/{filename}", target_size=(224, 224))
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

    x = layer.Dense(1024)(x)
    x = layer.Dense(1000)(x)

    output = layer.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=input, outputs=output)

    model.compile(tf.keras.optimizers.Adam(learning_rate=.001),
                  loss=tf.keras.losses.CosineSimilarity(),
                  metrics=['accuracy'])

    history_1 = model.fit(train_data, epochs=7, steps_per_epoch=len(train_data),
                          validation_data=test_data, validation_steps=len(test_data))

    predict(model, class_names)


def scheduler(epoch, lr):
    if epoch < 3:
        return lr
    elif epoch < 6:
        return .001
    else:
        return lr * tf.math.exp(-0.005)


def fashion_CNN():
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

    # TODO tune the kernel size of the convolutional layers
    x = base(inputs, training=False)
    x = layer.Conv2D(filters=64, kernel_size=7, strides=2, padding='same')(x)
    # x = incept(x, 64)
    x = sequence_a(x, 64, 1, 1)
    x = layer.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = sequence_a(x, 128, 1, 1)
    x = layer.Conv2D(filters=256, kernel_size=1, strides=1, padding='same')(x)
    x = sequence_a(x, 256, 1, 1)
    x = global_average_layer(x)
    x = layer.Flatten()(x)
    x = tf.keras.layers.Dense(2048)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(2048)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(2)(x)
    outputs = layer.Activation('sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)

    callback1 = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15,
        mode='min', baseline=None, restore_best_weights=True
    )
    callback2 = tf.keras.callbacks.LearningRateScheduler(scheduler)
    print(len(train_data))
    model.compile(tf.keras.optimizers.Adam(learning_rate=.01),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    history_1 = model.fit(train_data, epochs=20, steps_per_epoch=len(train_data),
                          validation_data=test_data, validation_steps=len(test_data), callbacks=[callback1, callback2])
    # predict(model, class_names)
    return model


# Make a function for preprocessing images
def preprocess_img(image, label, img_shape=28):
  image = tf.image.resize(image, [img_shape, img_shape]) # reshape to img_shape
  return tf.cast(image, tf.float32), label # return (float32_image, label) tuple


def get_dataset(name_of):
    (train_data, test_data), info = tfds.load(name_of, split=["train", "test"], with_info=True, shuffle_files=True,
                                              as_supervised=True)
    data = (train_data, test_data)

    return data, info


def item_type_CNN():
    data, info = get_dataset("fashion_mnist")
    labels = info.features["label"].names
    print(len(labels))

    # for image, label in data[0]:
    #     print(f"""
    #   Image shape: {image.shape}
    #   Image dtype: {image.dtype}
    #   Target class from Food101 (tensor form): {label}
    #   Class name (str form): {labels[label.numpy()]}
    #         """)

    # Map preprocessing function to training data (and paralellize)
    train_data = data[0].map(map_func=preprocess_img, num_parallel_calls=tf.data.AUTOTUNE).shuffle(buffer_size=1000)\
        .batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)
    train_data = train_data

    # Map prepreprocessing function to test data
    test_data = data[1].map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE).batch(32).prefetch(tf.data.AUTOTUNE)
    input = layer.Input(shape=(28, 28, 1))
    x = layer.Conv2D(32, (3, 3), activation="relu")(input)
    x = residual(x, 32, 1)
    x = residual(x, 32, 1)
    x = residual(x, 32, 1)
    x = layer.GlobalAveragePooling2D()(x)
    x = layer.Flatten()(x)
    x = layer.Dense(256, activation='relu')(x)
    output = layer.Dense(10, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=input, outputs=output)

    callback1 = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15,
        mode='min', baseline=None, restore_best_weights=True
    )
    callback2 = tf.keras.callbacks.LearningRateScheduler(scheduler)
    model.compile(tf.keras.optimizers.Adam(learning_rate=.01),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    history_1 = model.fit(train_data, epochs=30, steps_per_epoch=len(train_data),
                          validation_data=test_data, validation_steps=len(test_data), callbacks=[callback1, callback2])



# build_CNN()
# fashion_CNN()

# item_type_CNN()
