import pandas
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def preprocess(data):
    X = data['Job_offer'].to_numpy()
    y = data['Label'].to_numpy()
    

    X = np.expand_dims(X, 2)
    y = np.expand_dims(y, 2)
    return X, y

def model():
    model = keras.models.Sequential(
        [
            layers.LSTM(128, input_shape=(50, 1)),
            layers.Dense(5)
        ]
    )
    return model

if __name__ == "__main__":
    file = pandas.read_csv("../data/train_set.csv", delimiter = ",")
    print(file.head())
    X, y = preprocess(file)
    model = model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy()])
    print(X.shape)
    print(y.shape)
    history = model.fit(x=X, y=y)
    print(history)
