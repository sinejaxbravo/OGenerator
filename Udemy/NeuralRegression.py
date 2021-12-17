import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#
# model = tf.keras.Sequential([
#     tf.keras.Input(shape=(3,)),
#     tf.keras.layers.Dense(100, activation="relu"),
#     tf.keras.layers.Dense(100, activation="relu"),
#     tf.keras.layers.Dense(100, activation="relu"),
#     tf.keras.layers.Dense(1, activation=None)
# ])

X = tf.range(-1000,  1000, 4)
Y = X + 10

X_Train = X[:40]
X_Test = X[40:]
Y_Train = Y[:40]
Y_Test = Y[40:]
print(len(X_Train))

house_info = tf.constant(["bedroom", "bathroom", "garage"])
house_price = tf.constant([939700])

tf.random.set_seed(42)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation="relu", name="Input_Layer"),
    tf.keras.layers.Dense(100, activation="relu", name="Second_Layer"),
    tf.keras.layers.Dense(100, activation="relu", name="Third_Layer"),
    tf.keras.layers.Dense(10, name="Output_Layer")
], name="First_Model")

model.compile(loss=tf.keras.losses.mse,
              optimizer=tf.keras.optimizers.Adam(learning_rate=.01),
              metrics=["mse"]
              )
model.fit(X, Y, epochs=100)

z = model.predict([340])
tot = z[0][1] + z[0][2]+z[0][3] + z[0][4]+z[0][0]
tot /= 5

print(tot)
print(model.summary())
