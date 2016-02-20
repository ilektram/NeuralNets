import logging

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adadelta

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
    return (x_train, x_test, y_train, y_test)


x_train, x_test, y_train, y_test = train_test("train.csv")

logging.debug("SHAPES: IN Train [{}], Test [{}]".format(x_train.shape, x_test.shape))
logging.debug("SHAPES: OUT Train [{}], Test [{}]".format(y_train.shape, y_test.shape))


# Create NN for 2-layer unidimensional regression
model = Sequential()
model.add(Dense(60, input_shape=(x_train.shape[1],)))
model.add(Activation('sigmoid'))
model.add(Dense(output_dim=1))
model.add(Activation('softmax'))

#learning_rate = .1
#sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)

model.compile(loss='binary_crossentropy', optimizer=adadelta)

epochs = 10
model.fit(x_train, y_train, nb_epoch=epochs)
# score = model.evaluate(X_test, y_test, batch_size=16)

#predicted = model.predict(X_test)
#rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0))
#print("This NN's RMSE: {}".format(rmse))

#score = model.evaluate(X_test, y_test, show_accuracy=True, batch_size=200)

#print('Test score: {}'.format(score[0]))
#print('Test accuracy: {}'.format(score[1]))
