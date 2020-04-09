import tensorflow as tf
from keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Sequential

import matplotlib.pyplot as plotter
import numpy as np

from PIL import Image


def predict_image(model, image):
    return model.predict((np.asarray(Image.open(image).convert('L').resize((28, 28))) / 255.0)[np.newaxis, :, :])


def plot_all(fit, prop, optimizer):
    plotter.plot(fit.history['val_' + prop], label=optimizer)
    plotter.title(prop)
    plotter.ylabel('loss')
    plotter.xlabel('epochs')
    plotter.legend()


mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

optimizers = ['adam', 'nadam', 'rmsprop', 'sgd', 'adamax', 'adagrad']
# optimizers = ['adam']

for optimizer in optimizers:
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    fit = model.fit(train_images, train_labels, epochs=5, batch_size=128, validation_data=(test_images, test_labels))

    plot_all(fit, 'loss', optimizer)
    # print(predict_image(model, '5.png'))

plotter.show()
plotter.clf()
