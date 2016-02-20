import string

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD


# Load training data
def load_data(input_data):
    df_train = pd.read_csv(input_data, index_col="ID")
    df_train = df_train.dropna(axis=0, how='any', thresh=None, subset=None)
    # df_train = df_train.fillna(df_train.mean())
    df_train.drop(['v125', 'v56', 'v22'], inplace=True, axis=1, errors='ignore')

    letters = enumerate(string.ascii_uppercase, start=1)

    for index, letter in letters:
        df_train.replace(letter, index, inplace=True)

    train = df_train.as_matrix()

    return train

train_ndarray = load_data("train.csv")


split = np.split(train_ndarray, [0, 1], axis=1)
train_inp = split[1]
train_out = split[2]

print(train_inp.shape, train_out.shape, train_ndarray.shape)


# Convert data to correct dimensions
#pole_expected = pole_expected.reshape(len(pole_inp), 1)

# Split data to training & testing sets
#X_train, X_test, y_train, y_test = train_test_split(pole_inp,
#                                                    pole_expected,
#                                                    test_size=0.33,
#                                                    random_state=42)

# Create NN for 2-layer unidimensional regression
#model = Sequential()
#model.add(Dense(2, input_shape=(pole_inp.shape[1],)))
#model.add(Activation('linear'))
#model.add(Dense(1))
#model.add(Activation('linear'))

#learning_rate = .1
#sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)

#model.compile(loss='mean_squared_error', optimizer=sgd)

#epochs = 10
#model.fit(X_train, y_train, nb_epoch=epochs, batch_size=200)
# score = model.evaluate(X_test, y_test, batch_size=16)

#predicted = model.predict(X_test)
#rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0))
#print("This NN's RMSE: {}".format(rmse))

#score = model.evaluate(X_test, y_test, show_accuracy=True, batch_size=200)

#print('Test score: {}'.format(score[0]))
#print('Test accuracy: {}'.format(score[1]))
