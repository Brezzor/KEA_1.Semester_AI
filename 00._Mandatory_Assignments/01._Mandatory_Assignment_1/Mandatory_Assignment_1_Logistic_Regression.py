import marimo

__generated_with = "0.11.25"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Mandatory Assignment #1 Logistic Regression""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## For your Mandatory Assignment 1 you must do the following:""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ***1. obtain the cars.csv file from the course materials folder in the course materials repository***

        ***2. using the dataset found in the file, complete the tasks below***

        <s>***3. formulate a regression problem***</s>

        ***4. formulate a logistic regression problem***

        ***5. design and train a model for each of the regression and logistic regression problems***

        ***6. fine tune your models***

        ***7. answer your problem formulations***

        ***8. write a max two page research article (in Latex) with relevant visuals***
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Obtain the cars.csv file from the course materials folder in the course materials repository""")
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    return np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Import 'cars.csv', select the appropriate columns and make sure to tranform the '?' value to a NaN value.""")
    return


@app.cell
def _(pd):
    column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin']
    raw_cars_df = pd.read_csv("./cars.csv", usecols=column_names, na_values='?')
    raw_cars_df.head()
    return column_names, raw_cars_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Check for the number of NaN values and which column has missing values.""")
    return


@app.cell
def _(raw_cars_df):
    raw_cars_df.isna().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Using the dataset found in the file, complete the tasks below""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### - Formulate a logistic regression problem (Classification model)""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""***Using Horsepower, Displacement and Cylinders. Can we predict the cars Origin?***""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### - Design and train a model for the logistic regression problem""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Make a copy of the raw dataset. Impute and tranform the the copied dataset so the missing values, gets assigned an appropriate value.""")
    return


@app.cell
def _(column_names, raw_cars_df):
    from sklearn.impute import KNNImputer

    cars_df_1 = raw_cars_df.copy()

    imputer = KNNImputer(n_neighbors=1)

    imputer.fit(cars_df_1[column_names])
    return KNNImputer, cars_df_1, imputer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Transform dataset to fix missing values.""")
    return


@app.cell
def _(cars_df_1, column_names, imputer):
    cars_df_1[column_names] = imputer.transform(cars_df_1[column_names])
    cars_df_1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Check for missing values again, to see if the missing values has been fixed.""")
    return


@app.cell
def _(cars_df_1):
    cars_df_1.isna().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""To prepare the 'origin' for One Hot Encoding. We map the 'origin' to their appropriate country.""")
    return


@app.cell
def _(cars_df_1):
    origin_mapping = {1: 'USA', 2: 'Japan', 3: 'Europe'}
    cars_df_mapped = cars_df_1.copy()
    cars_df_mapped['origin'] = cars_df_1['origin'].map(origin_mapping)
    cars_df_mapped.tail()
    return cars_df_mapped, origin_mapping


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Here we fit the 'origin' column, to the One Hot Encoder.""")
    return


@app.cell
def _(cars_df_mapped):
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder()
    encoder.fit(cars_df_mapped[['origin']])
    return OneHotEncoder, encoder


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Here we transform the 'origin' using the encoder and get a matrix.""")
    return


@app.cell
def _(cars_df_mapped, encoder):
    origin_encoded = encoder.transform(cars_df_mapped[['origin']])
    origin_encoded
    return (origin_encoded,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Visualization of the matrix as an array.""")
    return


@app.cell
def _(origin_encoded):
    origin_encoded_array = origin_encoded.toarray()
    origin_encoded_array[:8]
    return (origin_encoded_array,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Check that the encoder has the correct categories.""")
    return


@app.cell
def _(encoder):
    encoder.categories_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Make the training and testing DataFrame.""")
    return


@app.cell
def _(cars_df_1, origin_encoded_array):
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    X = cars_df_1[['horsepower','cylinders','displacement']]
    y = origin_encoded_array

    min_max_scaler = MinMaxScaler()

    min_max_scaler.fit(X)
    X = min_max_scaler.transform(X)
    return MinMaxScaler, X, min_max_scaler, plt, train_test_split, y


@app.cell
def _(X, train_test_split, y):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    X_train[:5], y_train[:5]
    return X_test, X_train, y_test, y_train


@app.cell
def _():
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import models
    from tensorflow.keras import layers
    from tensorflow.keras import optimizers
    from tensorflow.keras import losses
    from tensorflow.keras import metrics
    from tensorflow.keras import activations
    return activations, keras, layers, losses, metrics, models, optimizers, tf


@app.cell
def _(activations, layers, losses, metrics, tf):
    model_1 = tf.keras.Sequential([
        layers.Input(shape=(3,)),
        layers.Dense(100, activation=activations.relu),
        layers.Dense(3)
    ])

    model_1_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model_1.compile(
        optimizer=model_1_optimizer, 
        loss=losses.MeanAbsoluteError(),
        metrics=[metrics.RootMeanSquaredError(),metrics.BinaryAccuracy()]
    )

    model_1.summary()
    return model_1, model_1_optimizer


@app.cell
def _(X_train, model_1, y_train):
    from utils import Timer

    with Timer():
        model_1_history_1 = model_1.fit(
            X_train,
            y_train,
            epochs=1000,
            verbose=0,
            shuffle=True,
            validation_split=0.2
        )
    return Timer, model_1_history_1


@app.cell
def _(model_1_history_1, pd):
    from tensorflow.keras.callbacks import History

    def show_history(hist: History):
        hist_df = pd.DataFrame(hist.history)
        hist_df['epoch'] = hist.epoch    
        return hist_df.tail()

    show_history(model_1_history_1)
    return History, show_history


@app.cell
def _(History, model_1_history_1, plt):
    def plot_loss(hist: History):
        plt.plot(hist.history['loss'], label='loss')
        plt.plot(hist.history['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('MAE [origin]')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_rsme(hist: History):
        plt.plot(hist.history['root_mean_squared_error'], label='root_mean_squared_error')
        plt.plot(hist.history['val_root_mean_squared_error'], label='val_root_mean_squared_error')
        plt.xlabel('Epoch')
        plt.ylabel('RSME [origin]')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_accuracy(hist: History):
        plt.plot(hist.history['binary_accuracy'], label='binary_accuracy')
        plt.plot(hist.history['val_binary_accuracy'], label='val_binary_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('ACC [origin]')
        plt.legend()
        plt.grid(True)
        plt.show()

    plot_loss(model_1_history_1), plot_rsme(model_1_history_1), plot_accuracy(model_1_history_1)
    return plot_accuracy, plot_loss, plot_rsme


@app.cell
def _(X_test, model_1, y_test):
    test_results = {}

    test_results['model_1'] = model_1.evaluate(
        X_test,
        y_test, verbose=0)
    return (test_results,)


@app.cell
def _(activations, layers, losses, metrics, optimizers, tf):
    model_2 = tf.keras.Sequential([
        layers.Input(shape=(3,)),
        layers.Dense(100, activation=activations.sigmoid),
        layers.Dense(3)
    ])

    model_2_optimizer = optimizers.Adam(learning_rate=0.001)

    model_2.compile(
        optimizer=model_2_optimizer, 
        loss=losses.MeanAbsoluteError(),
        metrics=[metrics.RootMeanSquaredError(),metrics.BinaryAccuracy()]
    )

    model_2.summary()
    return model_2, model_2_optimizer


@app.cell
def _(Timer, X_train, model_2, y_train):
    with Timer():
        model_2_history_1 = model_2.fit(
            X_train,
            y_train,
            epochs=1000,
            verbose=0,
            shuffle=True,
            validation_split=0.2
        )
    return (model_2_history_1,)


@app.cell
def _(model_2_history_1, show_history):
    show_history(model_2_history_1)
    return


@app.cell
def _(model_2_history_1, plot_accuracy, plot_loss, plot_rsme):
    plot_loss(model_2_history_1), plot_rsme(model_2_history_1), plot_accuracy(model_2_history_1)
    return


@app.cell
def _(X_test, model_2, test_results, y_test):
    test_results['model_2'] = model_2.evaluate(
        X_test,
        y_test, verbose=0)
    return


@app.cell
def _(MinMaxScaler, cars_df_1, origin_encoded_array):
    X_2 = cars_df_1[['mpg','horsepower','cylinders','displacement','weight','acceleration','model year']]
    y_2 = origin_encoded_array

    min_max_scaler_2 = MinMaxScaler()

    min_max_scaler_2.fit(X_2)
    X_2 = min_max_scaler_2.transform(X_2)
    return X_2, min_max_scaler_2, y_2


@app.cell
def _(X_2, train_test_split, y_2):
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2,y_2,test_size=0.2,random_state=42)
    X_train_2[:5], y_train_2[:5]
    return X_test_2, X_train_2, y_test_2, y_train_2


@app.cell
def _(activations, layers, losses, metrics, tf):
    model_3 = tf.keras.Sequential([
        layers.Input(shape=(7,)),
        layers.Dense(100, activation=activations.relu),
        layers.Dense(3)
    ])

    model_3_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model_3.compile(
        optimizer=model_3_optimizer, 
        loss=losses.MeanAbsoluteError(),
        metrics=[metrics.RootMeanSquaredError(),metrics.BinaryAccuracy()]
    )

    model_3.summary()
    return model_3, model_3_optimizer


@app.cell
def _(Timer, X_train_2, model_3, y_train_2):
    with Timer():
        model_3_history_1 = model_3.fit(
            X_train_2,
            y_train_2,
            epochs=1000,
            verbose=0,
            shuffle=True,
            validation_split=0.2
        )
    return (model_3_history_1,)


@app.cell
def _(model_3_history_1, show_history):
    show_history(model_3_history_1)
    return


@app.cell
def _(model_3_history_1, plot_accuracy, plot_loss, plot_rsme):
    plot_loss(model_3_history_1), plot_rsme(model_3_history_1), plot_accuracy(model_3_history_1)
    return


@app.cell
def _(X_test_2, model_3, test_results, y_test_2):
    test_results['model_3'] = model_3.evaluate(
        X_test_2,
        y_test_2, verbose=0)
    return


@app.cell
def _(test_results):
    test_results
    return


@app.cell
def _(pd, test_results):
    pd.DataFrame(test_results, index=['MAE','RSME','ACC']).T
    return


if __name__ == "__main__":
    app.run()
