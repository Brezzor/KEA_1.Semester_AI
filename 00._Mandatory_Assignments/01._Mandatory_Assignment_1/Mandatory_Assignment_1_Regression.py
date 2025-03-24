import marimo

__generated_with = "0.11.25"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Mandatory Assignment #1 Regression""")
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

        <s>***4. formulate a logistic regression problem***</s>

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
    mo.md(r"""Import 'cars.csv', select the appropriate columns and make sure to tranform the '?' value to a NaN value. """)
    return


@app.cell
def _(pd):
    column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin']
    raw_cars_df = pd.read_csv("./cars.csv", usecols=column_names, na_values='?')
    raw_cars_df
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
    mo.md(r"""### - Formulate a regression (Regression model) problem""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""***Using Displacement and Cylinders. Can we predict the cars Horsepower?***""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### - Design and train a model for each of the regression and logistic regression problems""")
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
    mo.md(r"""Here we make the encoded array into a DataFrame with the appropriate column names.""")
    return


@app.cell
def _(origin_encoded_array, pd):
    encoded_df = pd.DataFrame(origin_encoded_array, columns=['usa', 'japan', 'europe'])
    encoded_df.tail()
    return (encoded_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Here we drop the 'origin' column and and the new values to the DataFrame.""")
    return


@app.cell
def _(cars_df_mapped, encoded_df, pd):
    cars_df_copy = cars_df_mapped.drop(columns=['origin']).reset_index(drop=True)
    cars_df_2 = pd.concat([cars_df_copy, encoded_df], axis=1)
    cars_df_2.tail()
    return cars_df_2, cars_df_copy


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Check correlation for the DataFrame.""")
    return


@app.cell
def _(cars_df_2):
    cars_df_2.corr()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Make the training and testing DataFrame.""")
    return


@app.cell
def _(cars_df_2):
    cars_train_df = cars_df_2.sample(frac=0.8, random_state=0)
    cars_test_df = cars_df_2.drop(cars_train_df.index)
    cars_train_df, cars_test_df
    return cars_test_df, cars_train_df


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Make a Scatterplot to check for correlation in the training DataFrame.""")
    return


@app.cell
def _(cars_train_df):
    import seaborn as sns
    sns.pairplot(cars_train_df[['mpg', 'cylinders', 'displacement', 'weight', 'horsepower']], diag_kind='kde')
    return (sns,)


@app.cell
def _(mo):
    mo.md(r"""Check for the value distribution in the DataFrame.""")
    return


@app.cell
def _(cars_train_df):
    cars_train_df.describe().transpose()
    return


@app.cell
def _(cars_test_df, cars_train_df):
    cars_train_features = cars_train_df.copy()
    cars_test_features = cars_test_df.copy()

    cars_train_labels = cars_train_features.pop('horsepower')
    cars_test_labels = cars_test_features.pop('horsepower')
    return (
        cars_test_features,
        cars_test_labels,
        cars_train_features,
        cars_train_labels,
    )


@app.cell
def _(cars_train_df):
    cars_train_df.describe().transpose()[['mean', 'std']]
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
    dis_cyl = np.array(cars_train_features[['displacement','cylinders']])

    dis_cyl_normalizer = tf.keras.layers.Normalization(input_shape=[2,], axis=None)

    dis_cyl_normalizer.adapt(dis_cyl)
    return dis_cyl, dis_cyl_normalizer


@app.cell
def _(dis_cyl_normalizer, tf):
    horsepower_model = tf.keras.Sequential([
        dis_cyl_normalizer,
        tf.keras.layers.Dense(units=1)
    ])

    horsepower_model.summary()
    return (horsepower_model,)


@app.cell
def _(dis_cyl, horsepower_model):
    horsepower_model.predict(dis_cyl[:10])
    return


@app.cell
def _(horsepower_model, tf):
    horsepower_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_absolute_error')
    return


@app.cell
def _(cars_train_features, cars_train_labels, horsepower_model):
    import utils

    with utils.Timer():
        history_1 = horsepower_model.fit(
        cars_train_features[['displacement','cylinders']],
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
        #plt.ylim([0, 100])
        plt.xlabel('Epoch')
        plt.ylabel('Error [Horsepower]')
        plt.legend()
        plt.grid(True)
        plt.show()

    plot_loss(history_1)
    return plot_loss, plt


@app.cell
def _(cars_test_features, cars_test_labels, horsepower_model):
    test_results = {}

    test_results['horsepower_model'] = horsepower_model.evaluate(
        cars_test_features[['displacement','cylinders']],
        cars_test_labels, verbose=0)
    return (test_results,)


@app.cell
def _(horsepower_model, tf):
    def horsepower_model_predict():
        x = tf.linspace((32,0), 250, 251)
        y = horsepower_model.predict(x)
        return x, y

    hp_1_x, hp_1_y = horsepower_model_predict()
    return horsepower_model_predict, hp_1_x, hp_1_y


@app.cell
def _(cars_train_features, cars_train_labels, hp_1_x, hp_1_y, plt):
    def plot_displacement(x, y):
        plt.scatter(cars_train_features['displacement'], cars_train_labels, label='Data')
        plt.plot(x, y, color='k', label='Predictions')
        plt.xlabel('Displacement')
        plt.ylabel('Horsepower')
        plt.legend()
        plt.show()

    def plot_cylinders(x, y):
        plt.scatter(cars_train_features['cylinders'], cars_train_labels, label='Data')
        plt.plot(x, y, color='k', label='Predictions')
        plt.xlim(0,10)
        plt.xlabel('Cylinders')
        plt.ylabel('Horsepower')
        plt.legend()
        plt.show()

    plot_displacement(hp_1_x, hp_1_y), plot_cylinders(hp_1_x, hp_1_y)
    return plot_cylinders, plot_displacement


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
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
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
                    optimizer=tf.keras.optimizers.Adam(0.01))
      return model
    return (build_and_compile_model,)


@app.cell
def _(build_and_compile_model, dis_cyl_normalizer):
    dnn_horsepower_model = build_and_compile_model(dis_cyl_normalizer)
    return (dnn_horsepower_model,)


@app.cell
def _(cars_train_features, cars_train_labels, dnn_horsepower_model, utils):
    with utils.Timer():
        history_3 = dnn_horsepower_model.fit(
            cars_train_features[['displacement','cylinders']],
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
        x = tf.linspace((32,0), 250, 251)
        y = dnn_horsepower_model.predict(x)
        return x, y

    dnn_1_x, dnn_1_y = dnn_horsepower_model_predict()
    return dnn_1_x, dnn_1_y, dnn_horsepower_model_predict


@app.cell
def _(dnn_1_x, dnn_1_y, plot_cylinders, plot_displacement):
    plot_displacement(dnn_1_x, dnn_1_y), plot_cylinders(dnn_1_x, dnn_1_y)
    return


@app.cell
def _(
    cars_test_features,
    cars_test_labels,
    dnn_horsepower_model,
    test_results,
):
    test_results['dnn_horsepower_model'] = dnn_horsepower_model.evaluate(
        cars_test_features[['displacement','cylinders']], cars_test_labels,
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
    pd.DataFrame(test_results, index=['Mean absolute error [Horsepower]']).T
    return


@app.cell
def _(cars_test_features, cars_test_labels, dnn_model, plt):
    test_predictions = dnn_model.predict(cars_test_features).flatten()

    a = plt.axes(aspect='equal')
    plt.scatter(cars_test_labels, test_predictions)
    plt.xlabel('True Values [Horsepower]')
    plt.ylabel('Predictions [Horsepower]')
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
    plt.xlabel('Prediction Error [Horsepower]')
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

    test_results['dnn_model_reloaded'] = reloaded.evaluate(
        cars_test_features, cars_test_labels, verbose=0)
    return (reloaded,)


@app.cell
def _(pd, test_results):
    pd.DataFrame(test_results, index=['Mean absolute error [Horsepower]']).T
    return


if __name__ == "__main__":
    app.run()
