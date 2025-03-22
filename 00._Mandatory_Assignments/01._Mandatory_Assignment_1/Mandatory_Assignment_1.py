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
    raw_cars_dataset = pd.read_csv("./cars.csv", na_values='?')
    return (raw_cars_dataset,)


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


@app.cell
def _(raw_cars_dataset):
    cars_dataset_1 = raw_cars_dataset.copy()
    cars_dataset_1 = cars_dataset_1.dropna()
    return (cars_dataset_1,)


@app.cell
def _(cars_dataset_1):
    cars_dataset_1.info()
    return


@app.cell
def _(cars_dataset_1, pd):
    cars_dataset_2 = cars_dataset_1.copy()
    cars_dataset_2['origin'] = cars_dataset_2['origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
    cars_dataset_2 = pd.get_dummies(cars_dataset_2, columns=['origin'], prefix='', prefix_sep='')
    cars_dataset_2.tail()
    return (cars_dataset_2,)


@app.cell
def _(cars_dataset_2):
    cars_dataset_3 = cars_dataset_2.drop(columns=['car name'])
    cars_train_dataset = cars_dataset_3.sample(frac=0.8, random_state=0)
    cars_test_dataset = cars_dataset_3.drop(cars_train_dataset.index)
    cars_train_dataset.tail(), cars_test_dataset.tail()
    return cars_dataset_3, cars_test_dataset, cars_train_dataset


@app.cell
def _(cars_train_dataset):
    import seaborn as sns
    sns.pairplot(cars_train_dataset[['mpg', 'cylinders', 'displacement', 'weight']], diag_kind='kde')
    return (sns,)


@app.cell
def _(cars_train_dataset):
    cars_train_dataset.describe().transpose()
    return


@app.cell
def _(cars_test_dataset, cars_train_dataset):
    cars_train_features = cars_train_dataset.copy()
    cars_test_features = cars_test_dataset.copy()

    cars_train_labels = cars_train_features.pop('mpg')
    cars_test_labels = cars_test_features.pop('mpg')
    return (
        cars_test_features,
        cars_test_labels,
        cars_train_features,
        cars_train_labels,
    )


@app.cell
def _(cars_train_dataset):
    cars_train_dataset.describe().transpose()[['mean', 'std']]
    return


@app.cell
def _():
    import tensorflow as tf
    normalizer = tf.keras.layers.Normalization(axis=-1)
    return normalizer, tf


@app.cell
def _(cars_train_features, normalizer, np):
    normalizer.adapt(np.array(cars_train_features))
    normalizer.mean.numpy()
    return


@app.cell
def _(cars_train_features, normalizer, np):
    first = np.array(cars_train_features[:1], dtype=np.float32)

    with np.printoptions(precision=2, suppress=True):
      print('First example:', first)
      print()
      print('Normalized:', normalizer(first).numpy())
    return (first,)


@app.cell
def _(cars_train_features, np, tf):
    horsepower = np.array(cars_train_features['horsepower'])

    horsepower_normalizer = tf.keras.layers.Normalization(input_shape=[1,], axis=None)

    horsepower_normalizer.adapt(horsepower)
    return horsepower, horsepower_normalizer


@app.cell
def _(horsepower_normalizer, tf):
    horsepower_model = tf.keras.Sequential([
        horsepower_normalizer,
        tf.keras.layers.Dense(units=1)
    ])

    horsepower_model.summary()
    return (horsepower_model,)


@app.cell
def _(horsepower, horsepower_model):
    horsepower_model.predict(horsepower[:10])
    return


@app.cell
def _(horsepower_model, tf):
    horsepower_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        loss='mean_absolute_error')
    return


@app.cell
def _(cars_train_features, cars_train_labels, horsepower_model):
    import utils

    with utils.Timer():
        history_1 = horsepower_model.fit(
        cars_train_features['horsepower'],
        cars_train_labels,
        epochs=100,
        verbose=0,
        validation_split = 0.2)
    return history_1, utils


@app.cell
def _(history_1, pd):
    hist_1 = pd.DataFrame(history_1.history)
    hist_1['epoch'] = history_1.epoch
    hist_1.tail()
    return (hist_1,)


@app.cell
def _(history_1):
    import matplotlib.pyplot as plt

    def plot_loss(hist):
        plt.plot(hist.history['loss'], label='loss')
        plt.plot(hist.history['val_loss'], label='val_loss')
        plt.ylim([0, 10])
        plt.xlabel('Epoch')
        plt.ylabel('Error [MPG]')
        plt.legend()
        plt.grid(True)
        plt.show()

    plot_loss(history_1)
    return plot_loss, plt


@app.cell
def _(cars_test_features, cars_test_labels, horsepower_model):
    test_results = {}

    test_results['horsepower_model'] = horsepower_model.evaluate(
        cars_test_features['horsepower'],
        cars_test_labels, verbose=0)
    return (test_results,)


@app.cell
def _(horsepower_model, tf):
    def horsepower_model_predict():
        x = tf.linspace(0.0, 250, 251)
        y = horsepower_model.predict(x)
        return x, y

    hp_1_x, hp_1_y = horsepower_model_predict()
    return horsepower_model_predict, hp_1_x, hp_1_y


@app.cell
def _(cars_train_features, cars_train_labels, hp_1_x, hp_1_y, plt):
    def plot_horsepower(x, y):
        plt.scatter(cars_train_features['horsepower'], cars_train_labels, label='Data')
        plt.plot(x, y, color='k', label='Predictions')
        plt.xlabel('Horsepower')
        plt.ylabel('MPG')
        plt.legend()
        plt.show()

    plot_horsepower(hp_1_x, hp_1_y)
    return (plot_horsepower,)


@app.cell
def _(normalizer, tf):
    linear_model = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(units=1)
    ])
    return (linear_model,)


@app.cell
def _(cars_train_features, linear_model):
    linear_model.predict(cars_train_features[:10])
    return


@app.cell
def _(linear_model):
    linear_model.layers[1].kernel
    return


@app.cell
def _(linear_model, tf):
    linear_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        loss='mean_absolute_error')
    return


@app.cell
def _(cars_train_features, cars_train_labels, linear_model, utils):
    with utils.Timer():
        history_2 = linear_model.fit(
            cars_train_features,
            cars_train_labels,
            epochs=100,
            verbose=0,
            validation_split = 0.2)
    return (history_2,)


@app.cell
def _(history_2, plot_loss):
    plot_loss(history_2)
    return


@app.cell
def _(cars_test_features, cars_test_labels, linear_model, test_results):
    test_results['linear_model'] = linear_model.evaluate(
        cars_test_features, cars_test_labels, verbose=0)
    return


@app.cell
def _(tf):
    def build_and_compile_model(norm):
      model = tf.keras.Sequential([
          norm,
          tf.keras.layers.Dense(64, activation='relu'),
          tf.keras.layers.Dense(64, activation='relu'),
          tf.keras.layers.Dense(1)
      ])

      model.compile(loss='mean_absolute_error',
                    optimizer=tf.keras.optimizers.Adam(0.001))
      return model
    return (build_and_compile_model,)


@app.cell
def _(build_and_compile_model, horsepower_normalizer):
    dnn_horsepower_model = build_and_compile_model(horsepower_normalizer)
    return (dnn_horsepower_model,)


@app.cell
def _(cars_train_features, cars_train_labels, dnn_horsepower_model, utils):
    with utils.Timer():
        history_3 = dnn_horsepower_model.fit(
            cars_train_features['horsepower'],
            cars_train_labels,
            epochs=100,
            verbose=0,
            validation_split=0.2)
    return (history_3,)


@app.cell
def _(history_3, plot_loss):
    plot_loss(history_3)
    return


@app.cell
def _(dnn_horsepower_model, tf):
    def dnn_horsepower_model_predict():
        x = tf.linspace(0.0, 250, 251)
        y = dnn_horsepower_model.predict(x)
        return x, y

    dnn_1_x, dnn_1_y = dnn_horsepower_model_predict()
    return dnn_1_x, dnn_1_y, dnn_horsepower_model_predict


@app.cell
def _(dnn_1_x, dnn_1_y, plot_horsepower):
    plot_horsepower(dnn_1_x, dnn_1_y)
    return


@app.cell
def _(
    cars_test_features,
    cars_test_labels,
    dnn_horsepower_model,
    test_results,
):
    test_results['dnn_horsepower_model'] = dnn_horsepower_model.evaluate(
        cars_test_features['horsepower'], cars_test_labels,
        verbose=0)
    return


@app.cell
def _(build_and_compile_model, normalizer):
    dnn_model = build_and_compile_model(normalizer)
    dnn_model.summary()
    return (dnn_model,)


@app.cell
def _(cars_train_features, cars_train_labels, dnn_model, utils):
    with utils.Timer():
        history_4 = dnn_model.fit(
            cars_train_features,
            cars_train_labels,
            validation_split=0.2,
            verbose=0, epochs=100)
    return (history_4,)


@app.cell
def _(history_4, plot_loss):
    plot_loss(history_4)
    return


@app.cell
def _(cars_test_features, cars_test_labels, dnn_model, test_results):
    test_results['dnn_model'] = dnn_model.evaluate(cars_test_features, cars_test_labels, verbose=0)
    return


@app.cell
def _(pd, test_results):
    pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T
    return


@app.cell
def _(cars_test_features, cars_test_labels, dnn_model, plt):
    test_predictions = dnn_model.predict(cars_test_features).flatten()

    a = plt.axes(aspect='equal')
    plt.scatter(cars_test_labels, test_predictions)
    plt.xlabel('True Values [MPG]')
    plt.ylabel('Predictions [MPG]')
    lims = [0, 50]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()
    return a, lims, test_predictions


@app.cell
def _(cars_test_labels, plt, test_predictions):
    error = test_predictions - cars_test_labels
    plt.hist(error, bins=25)
    plt.xlabel('Prediction Error [MPG]')
    _ = plt.ylabel('Count')
    plt.show()
    return (error,)


@app.cell
def _(dnn_model):
    dnn_model.save('dnn_model.keras')
    return


@app.cell
def _(cars_test_features, cars_test_labels, test_results, tf):
    reloaded = tf.keras.models.load_model('dnn_model.keras')

    test_results['reloaded'] = reloaded.evaluate(
        cars_test_features, cars_test_labels, verbose=0)
    return (reloaded,)


@app.cell
def _(pd, test_results):
    pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T
    return


if __name__ == "__main__":
    app.run()
