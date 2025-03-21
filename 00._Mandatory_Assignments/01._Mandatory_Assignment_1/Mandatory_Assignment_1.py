import marimo

__generated_with = "0.11.25"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Mandatory Assignment #1""")
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

        ***3. formulate a regression problem***

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
    mo.md(r"""Import 'cars.csv'""")
    return


@app.cell
def _(pd):
    cars_ds = pd.read_csv("./cars.csv")
    return (cars_ds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Using the dataset found in the file, complete the tasks below""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### - Formulate a regression (Regression model) problem""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""***Using Displacement and Cylinders. Can we predict the cars Horsepower?***""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### - Formulate a logistic regression (Classification model) problem""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""***Using Displacement, Cylinders and Horsepower. Can we predict the manufacturer of the car?***""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### - Design and train a model for each of the regression and logistic regression problems""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Analyse the cars.csv dataset""")
    return


@app.cell
def _(cars_ds):
    cars_ds
    return


@app.cell
def _(cars_ds):
    cars_ds.info()
    return


@app.cell
def _(cars_ds):
    cars_ds.describe()
    return


@app.cell
def _(cars_ds):
    import matplotlib.pyplot as plt

    cars_ds.hist(figsize=(12,8))

    plt.show()
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The Horsepower column has six '?' values, that makes it an 'object'. One way to replace the question marks, is with a mean value.""")
    return


@app.cell
def _(cars_ds, np, pd):
    cars_ds_cleaned = cars_ds.copy()

    cars_ds_cleaned['horsepower'] = cars_ds_cleaned['horsepower'].replace('?', np.nan)

    cars_ds_cleaned['horsepower'] = pd.to_numeric(cars_ds_cleaned['horsepower'])

    cars_ds_cleaned.info()
    return (cars_ds_cleaned,)


@app.cell
def _(cars_ds_cleaned):
    from sklearn.impute import KNNImputer

    imputer = KNNImputer(n_neighbors=1)

    regression_columns = ['mpg','cylinders','displacement','horsepower','weight','acceleration','model year','origin']

    cars_ds_imputed = cars_ds_cleaned.copy()

    cars_ds_imputed[regression_columns] = imputer.fit_transform(cars_ds_imputed[regression_columns])

    cars_ds_imputed.info()
    return KNNImputer, cars_ds_imputed, imputer, regression_columns


@app.cell
def _(cars_ds_imputed, regression_columns):
    corr_matrix = cars_ds_imputed[regression_columns].corr()
    corr_matrix
    return (corr_matrix,)


@app.cell
def _(cars_ds_imputed, plt):
    from pandas.plotting import scatter_matrix
    corr_columns = ['cylinders','displacement','horsepower']
    scatter_matrix(cars_ds_imputed[corr_columns], figsize=(12,8))
    plt.show()
    return corr_columns, scatter_matrix


@app.cell
def _(cars_ds_imputed):
    from sklearn.model_selection import train_test_split

    car_regression_ds = cars_ds_imputed.drop(columns=['mpg','acceleration','model year','origin','car name'])

    X_regression = car_regression_ds[['cylinders','displacement','weight']]
    y_regression = car_regression_ds['horsepower']

    X_regression_train, X_regression_val, y_regression_train, y_regression_val = train_test_split(X_regression, y_regression, test_size=.8, random_state=42)
    return (
        X_regression,
        X_regression_train,
        X_regression_val,
        car_regression_ds,
        train_test_split,
        y_regression,
        y_regression_train,
        y_regression_val,
    )


@app.cell
def _(X_regression_train, X_regression_val):
    X_regression_train, X_regression_val
    return


@app.cell
def _(y_regression_train, y_regression_val):
    y_regression_train, y_regression_val
    return


@app.cell
def _():
    history = [] 
    return (history,)


@app.cell
def _():
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import models
    from tensorflow.keras import layers
    from tensorflow.keras import optimizers
    from tensorflow.keras import losses
    from tensorflow.keras import metrics
    return keras, layers, losses, metrics, models, optimizers, tf


@app.cell
def _(layers, models):
    model_1 = models.Sequential([
        layers.Input(shape=(3,)),
        layers.Dense(8, activation='sigmoid'),
        layers.Dense(10, activation='sigmoid'),
        layers.Dense(1, activation='sigmoid')
    ])
    return (model_1,)


@app.cell
def _(losses, metrics, model_1, optimizers):
    model_1.compile(
        optimizer=optimizers.SGD(learning_rate=0.001), 
        loss=losses.BinaryCrossentropy(),
        metrics=[metrics.BinaryAccuracy()]
    )
    return


@app.cell
def _(model_1):
    model_1.summary()
    return


@app.cell
def _(X_regression_train, history, model_1, y_regression_train):
    history.append(('Basic DNN', model_1.fit(
        X_regression_train,
        y_regression_train, 
        verbose=0, 
        epochs=1_000, 
        shuffle=True
    )))
    return


@app.cell
def _(X_regression_train, model_1, y_regression_train):
    model_1.evaluate(X_regression_train,y_regression_train)
    return


@app.cell
def _(X_regression_val, model_1, y_regression_val):
    model_1.evaluate(X_regression_val,y_regression_val)
    return


@app.cell
def _(layers, models):
    model_2 = models.Sequential([
        layers.Input(shape=(3,)),
        layers.Dense(8, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return (model_2,)


@app.cell
def _(losses, metrics, model_2, optimizers):
    model_2.compile(
        optimizer=optimizers.SGD(learning_rate=0.001), 
        loss=losses.BinaryCrossentropy(),
        metrics=[metrics.BinaryAccuracy()]
    )
    return


@app.cell
def _(X_regression_train, history, model_1, y_regression_train):
    history.append(('Basic relu DNN ', model_1.fit(
        X_regression_train,
        y_regression_train, 
        verbose=0, 
        epochs=1_000, 
        shuffle=True
    )))
    return


@app.cell
def _(X_regression_train, model_2, y_regression_train):
    model_2.evaluate(X_regression_train,y_regression_train)
    return


@app.cell
def _(X_regression_val, model_1, y_regression_val):
    model_1.evaluate(X_regression_val,y_regression_val)
    return


if __name__ == "__main__":
    app.run()
