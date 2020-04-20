
from keras.models import Sequential
from keras import layers

import numpy as np
import random
import math
import matplotlib.pyplot as plotter


def func(i):
    i = i % 31
    return ((i - 15) ** 2) / 100 - 4


def gen_sequence(seq_len=1000):
    seq = [math.cos(i / 2) + func(i) + random.normalvariate(0, 0.04) for i in range(seq_len)]
    return np.array(seq)


def draw_sequence(sequence):
    plotter.plot(range(len(sequence)), sequence)
    plotter.show()


def gen_data_from_sequence(seq_len=1006, lookback=10):
    seq = gen_sequence(seq_len)
    past = np.array([[[seq[j]] for j in range(i, i+lookback)] for i in range(len(seq) - lookback)])
    future = np.array([[seq[i]] for i in range(lookback, len(seq))])
    return past, future


data, res = gen_data_from_sequence()

dataset_size = len(data)
train_size = (dataset_size // 10) * 7
val_size = (dataset_size - train_size) // 2

train_data, train_res = data[:train_size], res[:train_size]
val_data, val_res = data[train_size:train_size+val_size], res[train_size:train_size+val_size]
test_data, test_res = data[train_size+val_size:], res[train_size+val_size:]

model = Sequential()
model.add(layers.GRU(32, recurrent_activation='hard_sigmoid', input_shape=(None, 1), return_sequences=True))
model.add(layers.LSTM(32, activation='relu', input_shape=(None, 1), return_sequences=True, dropout=0.2))
model.add(layers.GRU(32, input_shape=(None, 1), recurrent_dropout=0.2))
model.add(layers.Dense(1))

model.compile(optimizer='nadam', loss='mse')
history = model.fit(train_data, train_res, epochs=50, validation_data=(val_data, val_res))

loss = history.history['loss']
val_loss = history.history['val_loss']
plotter.plot(range(len(loss)), loss)
plotter.plot(range(len(val_loss)), val_loss)
plotter.title('loss/epoch')
plotter.xlabel('epoch')
plotter.ylabel('loss')
plotter.legend(['train', 'test'], loc='upper right')
plotter.show()

predicted_res = model.predict(test_data)
pred_length = range(len(predicted_res))
plotter.plot(pred_length, predicted_res)
plotter.plot(pred_length, test_res)
plotter.title('accuracy')
plotter.legend(['generated', 'predicted'], loc='upper right')
plotter.show()


draw_sequence(gen_sequence())
