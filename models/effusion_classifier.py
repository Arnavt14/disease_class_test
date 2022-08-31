from keras.models import Sequential
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.metrics import Precision, Recall, BinaryAccuracy
import matplotlib.pyplot as plt
import numpy as np
import cv2
# from constants import Effusion_constants
from constants import Effusion_constants

data = Effusion_constants.data


hist = None


def load_data():
    global data_iterator
    global batch
    data_iterator = data.as_numpy_iterator()
    batch = data_iterator.next()
    print(batch)

    return data_iterator, batch, data


def scaling_data(data):

    data = data.map(lambda x,y: (x/255, y))
    data.as_numpy_iterator().next()

    return data

train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)

# Splitting the data for training testing and validation

def test_train_split():
    global train
    global val
    global test
    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size + val_size).take(test_size)
    return train,val,test


 # Creating the CNN model by adding layers to it for binary classification

def create_model():
    global model
    model = Sequential()
    model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])


    return model


# Training the model
def train_model():
    # logdir='logs'
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    global hist
    hist = model.fit(train, epochs=1, validation_data=val)
    return hist

# Evaluating data

def eval():

    global pre
    global re
    global acc

    pre = Precision()
    re = Recall()
    acc = BinaryAccuracy()

    for batch in test.as_numpy_iterator():
        X, y = batch
        yhat = model.predict(X)
        pre.update_state(y, yhat)
        re.update_state(y, yhat)
        acc.update_state(y, yhat)

    print(pre.result(), re.result(), acc.result())

    return pre, re, acc


# Testing the model

def testing_model(img):
    img = img
    # plt.imshow(img)
    # plt.show()

    resize = tf.image.resize(img, (256, 256))
    # plt.imshow(resize.numpy().astype(int))
    # plt.show()

    yhat = model.predict(np.expand_dims(resize / 255, 0))

    if yhat > 0.5:  # because of sigmoid activation function
        print(f'Normal')
    else:
        print(f'Effusion')

