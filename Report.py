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

    # Fill categorical categories with NA value and convert them to the right
    # type
    for col in df_train.select_dtypes(include=['object']).columns:
        df_train[col] = df_train[col].fillna(value='NA', axis=0)
        df_train[col] = df_train[col].astype('category')

    # Fill the other columns with 0 as the fill value
    for col in df_train.select_dtypes(exclude=['category']).columns:
        df_train[col] = df_train[col].fillna(value=0, axis=0)

    old_length = df_train.shape[0]
    df_train = df_train.dropna(axis=0, how='any')
    row_diff = old_length - df_train.shape[0]
    logging.debug(
        "Dropped {} rows with NAs {:.1%}".format(
            row_diff,
            float(row_diff)/old_length
        )
    )
    return df_train


def categorical_to_front(input_df):
    cat_columns = list(input_df.select_dtypes(include=['category']).columns)

    logging.debug("Number of categorical columns: {}".format(len(cat_columns)))

    other_columns = list(input_df.select_dtypes(exclude=['category']).columns)

    new_column_order = cat_columns + other_columns
    train_df = input_df[new_column_order]

    return train_df


def train_test(input_data):
    """
    Splits data to training & testing sets
    Splits columns to input & output for training & testing set respectively
    input_data: a single csv file with an ID column
    """
    logging.info("Loading data from '{}'".format(input_data))
    train_df = load_data(input_data)

    # Reorder the columns, categorical go first
    train_df = categorical_to_front(train_df)

    # Temporary
    for col in train_df.select_dtypes(include=['category']).columns:
        train_df[col] = train_df[col].astype('category').cat.codes

    train_inp = train_df.drop('target', axis=1).as_matrix()
    train_out = train_df['target'].as_matrix()

    logging.debug(
        "Train 0s/1s: {:.2%} / {:.2%}".format(
            1.0 - np.average(train_out),
            np.average(train_out)
        )
    )

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
    model.add(Activation('linear'))
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
