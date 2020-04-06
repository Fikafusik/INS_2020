
from matplotlib import pyplot as plotter

from keras.layers import Input, Dense
from keras.models import Model

import pandas as pd
import numpy as np


def generate_data(amount):
    x_mu = 0
    x_sigma = np.sqrt(10)
    e_mu = 0
    e_sigma = np.sqrt(0.3)

    data = np.zeros((amount, 6))
    target = np.zeros(amount)

    for i in range(amount):
        x = x_sigma * np.random.randn() + x_mu
        e = e_sigma * np.random.randn() + e_mu
        target[i] = -x + e
        data[i, :] = np.cos(x) + e, np.sin(x) * x + e, np.sqrt(np.abs(x)) + e, x ** 2 + e, -np.abs(x) + 4, x - x ** 2 / 5 + e

    return data, target


def plot_loss(title, fit):
    plotter.plot(fit.history[title], 'y', label='train')
    plotter.plot(fit.history['val_' + title], 'g', label='validation')
    plotter.title(title)
    plotter.ylabel('loss')
    plotter.xlabel('epochs')
    plotter.legend()
    plotter.show()
    plotter.clf()


def plot_error(title, first, second):
    plotter.plot(first, 'r|')
    plotter.plot(second, 'g_')
    plotter.title(title)
    plotter.ylabel('val')
    plotter.legend()
    plotter.show()
    plotter.clf()


n = 300

train_data, train_target = generate_data(n)
validation_data, validation_target = generate_data(n // 10)

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data -= mean
train_data /= std
validation_data -= mean
validation_data /= std

input_dense = Input(shape=(6,), name='input')

encoded = Dense(64, activation='relu')(input_dense)
encoded = Dense(32, activation='relu')(encoded)
encoded = Dense(8, activation='relu', name="encode")(encoded)
encoder = Model(input_dense, encoded)
encoder.save('./encoder.h5')

decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(6, name='decode')(decoded)
decoder = Model(input_dense, decoded)
decoder.save('./decoder.h5')

predicted = Dense(64, activation='relu')(encoded)
predicted = Dense(32, activation='relu')(predicted)
predicted = Dense(16, activation='relu')(predicted)
predicted = Dense(1, name="predict")(predicted)
predictor = Model(input_dense, predicted)
predictor.save('./predictor.h5')

model = Model(input=input_dense, outputs=[predicted, decoded])
model.compile(optimizer='adam', loss='mse')
fit = model.fit(x=train_data, y=[train_target, train_data], epochs=100, batch_size=5,
                validation_data=(validation_data, [validation_target, validation_data]))

pd.DataFrame(train_data).to_csv("./train_data.csv")
pd.DataFrame(train_target).to_csv("./train_target.csv")

pd.DataFrame(validation_data).to_csv("./validation_data.csv")
pd.DataFrame(validation_target).to_csv("./validation_target.csv")

encoded_data = encoder.predict(validation_data)
pd.DataFrame(encoded_data).to_csv("./encoded_data.csv")

decoded_data = decoder.predict(validation_data)
pd.DataFrame(decoded_data).to_csv("./decoded_data.csv")

predicted_target = predictor.predict(validation_data)
pd.DataFrame(predicted_target).to_csv("./predicted_target.csv")

plot_loss('loss', fit)
plot_loss('predict_loss', fit)
plot_loss('decode_loss', fit)

plot_error('decoded_data - validation_data', decoded_data, validation_data)
plot_error('predicted_target - validation_target', predicted_target, validation_target)
