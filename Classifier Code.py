import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000: ] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

class_names[y_train[200]]

mat = (X_train[200,]*255).astype(int)
for i in range(mat.shape[0]):
    line = f""
    for j in range(mat.shape[1]):
        line += f"{mat[i,j]:>3} "
    print(line)
 #building a deep neural network
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(15, activation="relu"))
model.add(keras.layers.Dense(15, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))
#What is softmax:
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
#fitting the model
history = model.fit(X_train, y_train, epochs=15, validation_data=[X_valid, y_valid])
pd.DataFrame(history.history).plot(figsize=(7,7))
plt.grid()
model.summary()
model.evaluate(X_valid, y_valid)