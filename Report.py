import logging

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD, Adadelta
from keras.callbacks import EarlyStopping
from keras.utils import np_utils


logging.basicConfig(level=logging.DEBUG)


# Load training data
def load_data(input_data):
    """
    Loads data as ndarray
    input_data: a single csv file with an ID column
    """
    df_train = pd.read_csv(input_data, index_col="ID")

    for col in df_train.columns:
        if df_train[col].dtype == 'object':
            df_train[col] = df_train[col].astype('category').cat.codes

    df_train = df_train.dropna(axis=0, how='any', thresh=None, subset=None)

    train = df_train.as_matrix()
    return train


def train_test(input_data):
    """
    Splits data to training & testing sets
    Splits columns to input & output for training & testing set respectively
    input_data: a single csv file with an ID column
    """
    logging.info("Loading data from '{}'".format(input_data))
    train_ndarray = load_data(input_data)
    split = np.split(train_ndarray, [0, 1], axis=1)
    train_inp = split[2]
    train_out = split[1]
    x_train, x_test, y_train, y_test = train_test_split(train_inp,
                                                        train_out,
                                                        test_size=0.33,
                                                        random_state=42)
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    return (x_train, x_test, y_train, y_test)


x_train, x_test, y_train, y_test = train_test("train.csv")


logging.debug("SHAPES: IN Train [{}], Test [{}]".format(x_train.shape, x_test.shape))
logging.debug("SHAPES: OUT Train [{}], Test [{}]".format(y_train.shape, y_test.shape))


# Create NN for 2-layer unidimensional regression

# learning_rate = .1
# sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)

batch = len(x_train)
epochs = 100


def nn_model(x_train, y_train, epochs, batch, error_fun):

    model = Sequential()

    model.add(Dense(128, input_shape=(x_train.shape[1],)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(output_dim=2))
    model.add(Activation('softmax'))
    model.compile(class_mode='binary', loss='binary_crossentropy', optimizer=error_fun)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=0,
        mode='auto'
    )

    model.fit(
        x_train,
        y_train,
        nb_epoch=epochs,
        batch_size=batch,
        validation_split=0.1,
        show_accuracy=True,
        callbacks=[early_stopping]
    )

    return model

model = nn_model(x_train, y_train, epochs, batch, adadelta)

predicted = model.predict(x_test)
rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0))
print("This NN's RMSE: {}".format(rmse))

score = model.evaluate(x_test, y_test, show_accuracy=True, batch_size=batch)

print('Test score: {}'.format(score[0]))
print('Test accuracy: {}'.format(score[1]))
