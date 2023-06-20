from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from random import randint, random

import numpy as np
from absl import app
from keras.models import Sequential, Model, load_model
from keras.layers import Dense
from keras.layers import Input
import pandas as pd
from keras.optimizer_v1 import SGD
from keras.utils.vis_utils import plot_model
from tensorflow.python.keras.backend import expand_dims
import tensorflow as tf


def create_win_data(num):
    dataset = []
    for i in range(num):
        data = [randint(10, 25), randint(80, 160), random(), random(), randint(0, 4), randint(0, 3)]
        dataset.append(data)
    y = []
    for i in range(num):
        data = [0, 0, 1, 1]
        y.append(data)

    return pd.DataFrame(dataset), y


def create_loss_data(num):
    dataset = []
    for i in range(num):
        data = [randint(-1, 10), randint(0, 80), random(), random(), randint(0, 4), randint(0, 3)]
        dataset.append(data)
    y = []
    for i in range(num):
        data = [1, 1, 0, 0]
        y.append(data)

    return np.asarray(dataset).astype('float32'), np.asarray(y).astype('float32')

def get_data():
    df = pd.read_csv(r'fcpa_dataset_3.csv', index_col=0)
    X, y, money = df.iloc[:, :-2], df.iloc[:, -2:-1], df.iloc[:, -1:]

    X = tf.expand_dims(np.asarray(X).astype('float32'), axis=-1)
    y = np.asarray(y).astype('float32')
    money = np.asarray(money).astype('float32')

    y_train = np.zeros((y.size, 4))
    for i in range(len(y)):
        if y[i] == 0:
            y_train[i] = [1, 0, 0, 0]
        elif y[i] == 1:
            y_train[i] = [0, 1, 0, 0]
        elif y[i] == 2:
            y_train[i] = [0, 0, 1, 0]
        elif y[i] == 3:
            y_train[i] = [0, 0, 0, 1]

    # print(y_train)
    for i in range(len(money)):
        if money[i] > 0:
            for j in range(len(y_train[i])):
                if y_train[i][j] == 1:
                    y_train[i][j] = max(money[i] / 20000, 0.5)
                else:
                    y_train[i][j] = 0
        elif money[i] < -10000:
            for j in range(len(y_train[i])):
                if y_train[i][j] == 1:
                    y_train[i][j] = 0
                else:
                    y_train[i][j] = 0.7
        else:
            for j in range(len(y_train[i])):
                if y_train[i][j] == 1:
                    y_train[i][j] = 0.7
                else:
                    y_train[i][j] = 0.5
    return X, y_train
def main(_):

    # X, y_train = get_data()
    Xloss, yloss = create_loss_data(5000)
    Xwin, ywin = create_win_data(5000)

    X = np.concatenate((Xloss, Xwin))
    y_train = np.concatenate((yloss, ywin))

    # print(X)
    # print(y_train)
    model = load_model('fcpa_nn_with_fold')

    # print(X[1])
    # model = Sequential()
    # model.add(Dense(12, input_shape=(X.shape[1],), activation="relu"))
    # model.add(Dense(8, activation="relu"))
    # model.add(Dense(4, activation='softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam')
    # print(model.summary())
    # plot_model(model, to_file='img/model_plot.png', show_shapes=True, show_layer_names=True)

    history = model.fit(X, y_train, epochs=200, batch_size=50)

    y_pred = model.predict(tf.expand_dims(X[0], axis=0))
    print(y_pred)
    model.save('fcpa_nn_dummy')


if __name__ == "__main__":
    app.run(main)
