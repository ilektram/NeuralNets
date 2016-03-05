import logging
from pprint import pformat

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adagrad
from keras.callbacks import EarlyStopping
from keras.utils import np_utils


logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)-8s %(message)s'
)


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
        df_train[col] = df_train[col].fillna(value=-1, axis=0)

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


def categorical_analysis(input_data):
    categories = []
    for col in input_data.columns:
        if str(input_data[col].dtype) == "category":
            cat = {
                "col_lbl": col,
                "cat_count": input_data[col].cat.categories.shape[0]
            }
            categories.append(cat)
    return categories


def convert_category_to_columns(input_data, column_name):
    if not isinstance(input_data, pd.DataFrame):
        raise TypeError("Input data must be a Pandas DataFrame")
    if str(input_data[column_name].dtype) != "category":
        raise RuntimeError("Can only run this on categorical columns")
    categories = input_data[column_name].cat.categories

    for cat in categories:
        new_col_name = "{col}_{cat}".format(col=column_name, cat=cat)
        input_data[new_col_name] = np.where(
            input_data[column_name] == cat,
            1,
            0
        )


def convert_categories_to_columns(input_data, cat_thres=130):
    cols_to_remove = []
    for col in input_data.columns:
        if str(input_data[col].dtype) == 'category':
            if input_data[col].cat.categories.shape[0] < cat_thres:
                convert_category_to_columns(input_data, col)
                cols_to_remove.append(col)
    input_data.drop(cols_to_remove, axis=1, inplace=True)


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
    logging.debug(
        "Categories and label counts: {}".format(
            pformat(categorical_analysis(train_df))
        )
    )

    convert_categories_to_columns(train_df)

    logging.debug(train_df.get_dtype_counts())

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
adagrad = Adagrad()

# batch = len(x_train)
batch = 1024

epochs = 100


def nn_model(x_train, y_train, epochs, batch, error_fun):

    model = Sequential()

    model.add(Dense(128, input_shape=(x_train.shape[1],)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

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

model = nn_model(x_train, y_train, epochs, batch, adagrad)

predicted = model.predict(x_test)
logging.info("Predicted 0s/1s: {:.2%} {:.2%}".format(np.average(predicted[:, 0]), np.average(predicted[:, 1])))
print(predicted, y_test)
rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0))
print("This NN's RMSE: {}".format(rmse))

score = model.evaluate(x_test, y_test, show_accuracy=True, batch_size=batch)

print('Test score (log loss): {}'.format(score[0]))
print('Test accuracy: {}'.format(score[1]))
