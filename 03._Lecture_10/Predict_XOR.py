import marimo

__generated_with = "0.11.12"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""# Predict XOR using Classification and Regression""")
    return


@app.cell
def _(mo):
    mo.md(r"""Import numpy and sklearn.neural_network""")
    return


@app.cell
def _():
    import numpy as np
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    return MLPClassifier, MLPRegressor, np


@app.cell
def _(mo):
    mo.md(r"""Setup x and y""")
    return


@app.cell
def _(np):
    x = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,1,1,0])
    return x, y


@app.cell
def _(mo):
    mo.md(r"""Train model to Classify XOR""")
    return


@app.cell
def _(MLPClassifier, x, y):
    mlpc_cla = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
    mlpc_cla.fit(x,y)
    return (mlpc_cla,)


@app.cell
def _(mo):
    mo.md(
        """
        Show prediction

        """
    )
    return


@app.cell
def _(mlpc_cla, x):
    mlpc_cla.predict(x)
    return


@app.cell
def _(mo):
    mo.md(r"""Train model to do Regression on XOR""")
    return


@app.cell
def _(MLPRegressor, x, y):
    mlpc_reg = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000)
    mlpc_reg.fit(x,y)
    return (mlpc_reg,)


@app.cell
def _(mo):
    mo.md(r"""Show prediction""")
    return


@app.cell
def _(mlpc_reg, x):
    mlpc_reg.predict(x)
    return


if __name__ == "__main__":
    app.run()
