import marimo

__generated_with = "0.11.19"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    from sklearn import datasets
    import matplotlib.pyplot as plt
    import utils
    return datasets, np, plt, utils


@app.cell
def _(datasets, np):
    np.random.seed(0)

    m = 1_000
    split_train = int(m * 0.7)
    split_val = int(m * 0.15 + split_train)
    split_test = int(m * 0.15 + split_val)

    X, y = datasets.make_moons(
        n_samples=m, 
        noise=0.1, 
        random_state=0
    )

    X_train, y_train = X[:split_train], y[:split_train]
    X_val, y_val = X[split_train:split_val], y[split_train:split_val]
    X_test, y_test = X[split_val:split_test], y[split_val:split_test]
    return (
        X,
        X_test,
        X_train,
        X_val,
        m,
        split_test,
        split_train,
        split_val,
        y,
        y_test,
        y_train,
        y_val,
    )


@app.cell
def _(X_test, X_train, X_val, split_test, split_train, split_val):
    print('Splits:', split_train, split_val, split_test)
    print('Lenghths:', len(X_train), len(X_val), len(X_test))
    return


@app.cell
def _(X, plt, y):
    colors = ['blue' if label == 1 else 'red' for label in y]
    plt.scatter(X[:,0], X[:,1], color=colors)
    plt.show()
    print(y.shape, X.shape)
    return (colors,)


@app.cell
def _(X_train, y_train):
    X_train[0], y_train[0]
    return


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
    model1 = models.Sequential([
        layers.Flatten(input_shape=(2,)),
        layers.Dense(8, activation='sigmoid'),
        layers.Dense(10, activation='sigmoid'),
        layers.Dense(1, activation='sigmoid')
    ])
    return (model1,)


@app.cell
def _(losses, metrics, model1, optimizers):
    model1.compile(
        optimizer=optimizers.SGD(learning_rate=0.001), 
        loss=losses.BinaryCrossentropy(),
        metrics=[metrics.BinaryAccuracy()]
    )
    return


@app.cell
def _(model1):
    model1.summary()
    return


@app.cell
def _(X_train, model1, utils, y_train):
    with utils.Timer():
        history1 = model1.fit(
            X_train, 
            y_train, 
            verbose=0, 
            epochs=1_000, 
            shuffle=True
        )
    return (history1,)


@app.cell
def _(X_train, model1, y_train):
    model1.evaluate(X_train, y_train)
    return


@app.cell
def _(X_val, model1, y_val):
    model1.evaluate(X_val, y_val)
    return


@app.cell
def _(X, model1, utils, y):
    utils.plot_decision_boundary(X, y, model1, cmap='RdBu')
    return


@app.cell
def _(history1, plt):
    plt.plot([i for i in range(1_000)], history1.history['loss'])
    return


@app.cell
def _(X_train, model1, utils, y_train):
    with utils.Timer():
        history2 = model1.fit(
            X_train, 
            y_train, 
            verbose=0, 
            epochs=3_000, 
            shuffle=True
        )
    return (history2,)


@app.cell
def _(X_train, model1, y_train):
    model1.evaluate(X_train, y_train)
    return


@app.cell
def _(X_val, model1, y_val):
    model1.evaluate(X_val, y_val)
    return


@app.cell
def _(history2, plt):
    plt.plot([i for i in range(3_000)], history2.history['loss'])
    return


@app.cell
def _(X, model1, utils, y):
    utils.plot_decision_boundary(X, y, model1, cmap='RdBu')
    return


@app.cell
def _(layers, models):
    model2 = models.Sequential([
        layers.Flatten(input_shape=(2,)),
        layers.Dense(8, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return (model2,)


@app.cell
def _(losses, metrics, model2, optimizers):
    model2.compile(
        optimizer=optimizers.SGD(learning_rate=0.001), 
        loss=losses.BinaryCrossentropy(),
        metrics=[metrics.BinaryAccuracy()]
    )
    return


@app.cell
def _(X_train, model2, utils, y_train):
    with utils.Timer():
        history3 = model2.fit(
            X_train, 
            y_train, 
            verbose=0, 
            epochs=1_000, 
            shuffle=True
        )
    return (history3,)


@app.cell
def _(X_train, model2, y_train):
    model2.evaluate(X_train, y_train)
    return


@app.cell
def _(X_val, model2, y_val):
    model2.evaluate(X_val, y_val)
    return


@app.cell
def _(history3, plt):
    plt.plot([i for i in range(1_000)], history3.history['loss'])
    return


@app.cell
def _(layers, models):
    model3 = models.Sequential([
        layers.Flatten(input_shape=(2,)),
        layers.Dense(8, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return (model3,)


@app.cell
def _(losses, metrics, model3, optimizers):
    model3.compile(
        optimizer=optimizers.Adam(learning_rate=0.001), 
        loss=losses.BinaryCrossentropy(),
        metrics=[metrics.BinaryAccuracy()]
    )
    return


@app.cell
def _(X_train, model3, utils, y_train):
    with utils.Timer():
        history4 = model3.fit(
            X_train, 
            y_train, 
            verbose=0, 
            epochs=1_000, 
            shuffle=True
        )
    return (history4,)


@app.cell
def _(X_train, model3, y_train):
    model3.evaluate(X_train, y_train)
    return


@app.cell
def _(X_val, model3, y_val):
    model3.evaluate(X_val, y_val)
    return


@app.cell
def _(history4, plt):
    plt.plot([i for i in range(1_000)], history4.history['loss'])
    return


@app.cell
def _(X, model3, utils, y):
    utils.plot_decision_boundary(X, y, model3, cmap='RdBu')
    return


@app.cell
def _(X_test, model3, y_test):
    model3.evaluate(X_test, y_test)
    return


@app.cell
def _(X_test, model3, utils, y_test):
    utils.plot_decision_boundary(X_test, y_test, model3, cmap='RdBu')
    return


if __name__ == "__main__":
    app.run()
