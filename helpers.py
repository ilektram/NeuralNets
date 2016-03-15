"""
Helper fns for easier data analysis
"""

import pandas as pd
from bokeh.charts import Histogram, show
from bokeh.plotting import figure, show
from sklearn.metrics import roc_curve, precision_recall_curve


def load_data(input_data):
    """
    Loads data as ndarray
    input_data: a single csv file with an ID column
    """
    df_train = pd.read_csv(input_data, index_col="ID")
    mark_categorical_columns(df_train)
    fill_non_categorical_columns(df_train)
    return df_train


def mark_categorical_columns(input_df, fill_value='NA', inplace=True):
    """ Change the dtype of the columns whose type is "object" to categorical and also fill in NA values
    :param input_df: a Pandas DataFrame to be used as input
    :param fill_value: Value to be used to fill NAs. Doesn't fill NA if None is given
    :param inplace: Does the operation in place (altering input_df)
    :return: Altered copy (if inplace=True) or input_df otherwise
    """
    if not inplace:
        res_df = input_df.copy(deep=True)
    else:
        res_df = input_df

    for col in res_df.select_dtypes(include=['object']).columns:
        if fill_value is not None:
            res_df[col] = res_df[col].fillna(value=fill_value, axis=0)
        res_df[col] = res_df[col].astype('category')

    return res_df


def fill_non_categorical_columns(input_df, value, inplace=True):
    if not inplace:
        res_df = input_df.copy(deep=True)
    else:
        res_df = input_df

    # Fill the other columns with 0 as the fill value
    for col in res_df.select_dtypes(exclude=["category"]).columns:
        res_df[col] = res_df[col].fillna(value=-1, axis=0)

    return res_df


def add_column_with_NA_count(input_df, inplace=True):
    if not inplace:
        res_df = input_df.copy(deep=True)
    else:
        res_df = input_df

    res_df.loc[:, 'NA_count'] = res_df.isnull().sum(axis=1)

    return res_df


def histogram_for_predicted_proba(y_expected, y_pred_proba, title):
    y_compare = pd.DataFrame({"target": y_expected.ravel(), "prediction": y_pred_proba.ravel()})
    print y_compare.describe()
    p = Histogram(y_compare[y_compare["target"] == 0], 'prediction', title=title)
    show(p)


def histogram_for_training_loss_accuracy(hist_list):
    #TODO: Not Complete yet, please finish me
    epochs = len(hist_list[0].history['acc'])
    p = figure(title="Model Performance (Training Set)", plot_width=600, plot_height=600)

    data = pd.DataFrame(columns=['epoch', 'loss', 'acc', 'fold'])

    for idx, fold_data in enumerate(hist_list):
        fold_df = pd.DataFrame(
            data=fold_data.history,
            columns=['loss', 'acc']
        )
        fold_df['fold'] = idx
        fold_df['epoch'] = fold_df.index

        data.append(fold_df)

    for hist_obj in hist_list:
        p.line(x=range(0, epochs), y=hist_obj.history['loss'],
               color="firebrick", line_width=4, legend="Loss")
        p.line(x=range(0, epochs), y=hist_list.history['acc'],
               color="navy", line_width=4, legend="Accuracy")

    p.legend.orientation = "bottom_left"
    p.xaxis.axis_label = "Epoch"


def epochs_perf_plot(hist):
    """
    Create plot of model performance by epoch
    input: nn history object, # epochs
    returns bokeh line plot
    """
    epochs = len(hist.history['acc'])
    p = figure(title="Model Performance (Training Set)", plot_width=600, plot_height=600)

    p.line(x=range(0, epochs), y=hist.history['loss'],
           color="firebrick", line_width=4, legend="Loss")
    p.line(x=range(0, epochs), y=hist.history['acc'],
           color="navy", line_width=4, legend="Accuracy")

    p.legend.orientation = "bottom_left"
    p.xaxis.axis_label = "Epoch"

    show(p)


def plot_ROC(y_test, predicted):
    """
    Plots the ROC curve for the model
    input: true label vector of the data, predicted value vector
    """
    # Get true positive rate & false positive rate
    fpr, tpr, thresholds = roc_curve(y_test, predicted)
    # Plot ROC
    p = figure(title="Model Metrics (ROC)", plot_width=600, plot_height=600)

    p.line(x=fpr, y=tpr, color="navy", line_width=4)
    p.xaxis.axis_label = "False Positive Rate"
    p.yaxis.axis_label = "True Positive Rate"

    show(p)


def plot_PRC(y_test, predicted):
    """
    Plots the PRC curve for the model
    input: true label vector of the data, predicted value vector
    """
    # Get Precision & Recall metrics on test set
    precision, recall, thresholds = precision_recall_curve(y_test, predicted)
    # Plot PRC
    p = figure(title="Model Metrics (PRC)", plot_width=600, plot_height=600)

    p.line(x=recall, y=precision, color="firebrick", line_width=4)
    p.xaxis.axis_label = "Recall"
    p.yaxis.axis_label = "Precision"

    show(p)


def convert_cat_to_plain_numerical(df, col_lbl):
    df.loc[:, col_lbl] = df.loc[:, col_lbl].cat.codes
