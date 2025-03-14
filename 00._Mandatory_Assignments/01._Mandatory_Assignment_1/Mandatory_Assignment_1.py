import marimo

__generated_with = "0.11.19"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""# Mandatory Assignment #1""")
    return


@app.cell
def _():
    import marimo as mo

    import numpy as np
    import pandas as pd
    return mo, np, pd


@app.cell
def _(mo):
    mo.md(r"""Import 'cars.csv'""")
    return


@app.cell
def _(pd):
    cars = pd.read_csv("./cars.csv")
    cars.head()
    return (cars,)


if __name__ == "__main__":
    app.run()
