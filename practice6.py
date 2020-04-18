
import gens
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from matplotlib import pyplot as plotter
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle


def gen_data(size=500, img_size=50):
    c1 = size // 2
    c2 = size - c1

    label_c1 = np.full([c1, 1], 'Square') # 0 means Square
    data_c1 = np.array([gens.gen_rect(img_size) for i in range(c1)])
    label_c2 = np.full([c2, 1], 'Circle') # 1 means Circle
    data_c2 = np.array([gens.gen_empty_circle(img_size) for i in range(c2)])

    data = np.vstack((data_c1, data_c2))
    label = np.vstack((label_c1, label_c2))

    return data, label


n = 500
train_data, train_target = gen_data(n)
test_data, test_target = gen_data(n // 10)

train_data, train_target = shuffle(train_data, train_target)

train_data = train_data.reshape([-1, 50, 50, 1])
test_data = test_data.reshape([-1, 50, 50, 1])

encoder = LabelEncoder()
train_target = encoder.fit_transform(train_target)
test_target = encoder.fit_transform(test_target)

train_target = keras.utils.to_categorical(train_target, 2)
test_target = keras.utils.to_categorical(test_target, 2)

input_shape = (50, 50, 1)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

fit = model.fit(train_data, train_target, batch_size=10, epochs=10, verbose=1,
                validation_data=(test_data, test_target))

score = model.evaluate(test_data, test_target, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


def plot_loss(title, fit):
    plotter.plot(fit.history[title], 'y', label='train')
    plotter.plot(fit.history['val_' + title], 'g', label='validation')
    plotter.title(title)
    plotter.ylabel('loss')
    plotter.xlabel('epochs')
    plotter.legend()
    plotter.show()
    plotter.clf()


plot_loss('loss', fit)
plot_loss('acc', fit)
