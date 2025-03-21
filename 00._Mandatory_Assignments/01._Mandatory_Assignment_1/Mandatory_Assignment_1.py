import marimo

__generated_with = "0.11.22"
app = marimo.App()


@app.cell
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


@app.cell
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
    mo.md(r"""### Formulate a regression (Regression model) problem""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""***Using MPG and Displacement. Can we predict the cars Horsepower?***""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Formulate a logistic regression (Classification model) problem""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""***Using MPG, Displacement and Horsepower. Can we predict the manufacturer of the car?***""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Design and train a model for each of the regression and logistic regression problems""")
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
    cars_ds_cleaned['horsepower'].replace('?', np.nan, inplace=True)
    cars_ds_cleaned['horsepower'] = pd.to_numeric(cars_ds_cleaned['horsepower'])
    cars_ds_cleaned
    return (cars_ds_cleaned,)


@app.cell
def _(cars_ds_cleaned):
    from sklearn.impute import KNNImputer
    imputer = KNNImputer(n_neighbors=2)
    cars_ds_imputed = cars_ds_cleaned.copy()
    cars_ds_imputed[["horsepower"]] = imputer.fit_transform(cars_ds_imputed[["horsepower"]])
    cars_ds_imputed
    return KNNImputer, cars_ds_imputed, imputer


if __name__ == "__main__":
    app.run()
