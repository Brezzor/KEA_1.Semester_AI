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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Check for the value distribution in the DataFrame.""")
    return


@app.cell
def _(cars_train_df):
    cars_train_df.describe().transpose()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Make the features DataFrame, with the values we can test with, and the labels Series, with the values we want to predict for.""")
    return


@app.cell
def _(cars_test_df, cars_train_df, pd):
    cars_train_features = cars_train_df.copy()
    cars_test_features = cars_test_df.copy()

    cars_train_labels = pd.DataFrame(cars_train_features.pop('usa'))
    cars_train_labels['japan'] = (cars_train_features.pop('japan'))
    cars_train_labels['europe'] = (cars_train_features.pop('europe'))

    cars_test_labels = pd.DataFrame(cars_test_features.pop('usa'))
    cars_test_labels['japan'] = (cars_test_features.pop('japan'))
    cars_test_labels['europe'] = (cars_test_features.pop('europe'))

    cars_train_features, cars_test_features, cars_train_labels, cars_test_labels
    return (
        cars_test_features,
        cars_test_labels,
        cars_train_features,
        cars_train_labels,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Check to see the DataFrame Standard Deviation.""")
    return


@app.cell
def _(cars_train_df):
    cars_train_df.describe().transpose()[['mean', 'std']]
    return


@app.cell
def _(cars_train_features, np):
    import tensorflow as tf

    hors_dis_cyl_array = np.array(cars_train_features[['horsepower','displacement','cylinders']])

    hors_dis_cyl_array[:5]
    return hors_dis_cyl_array, tf


@app.cell
def _(hors_dis_cyl_array, tf):
    hors_dis_cyl_normalizer = tf.keras.layers.Normalization(axis=None)

    hors_dis_cyl_normalizer.adapt(hors_dis_cyl_array)
    return (hors_dis_cyl_normalizer,)


@app.cell
def _(hors_dis_cyl_normalizer, tf):
    model_1 = tf.keras.Sequential([
        hors_dis_cyl_normalizer,
        tf.keras.layers.Dense(8, activation='sigmoid'),
        tf.keras.layers.Dense(3)
    ])

    model_1_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    model_1.compile(
        optimizer=model_1_optimizer, 
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    model_1.summary()
    return model_1, model_1_optimizer


@app.cell
def _(cars_train_features, cars_train_labels, model_1):
    from utils import Timer

    with Timer():
        model_1_history_1 = model_1.fit(
            cars_train_features[['horsepower','displacement','cylinders']],
            cars_train_labels,
            epochs=100,
            verbose=0,
            validation_split=0.2
        )
    return Timer, model_1_history_1


@app.cell
def _(model_1_history_1):
    import matplotlib.pyplot as plt

    def plot_loss(hist):
        plt.plot(hist.history['val_binary_accuracy'], label='val_binary_accuracy')
        plt.plot(hist.history['binary_accuracy'], label='binary_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Error [origin]')
        plt.legend()
        plt.grid(True)
        plt.show()

    plot_loss(model_1_history_1)
    return plot_loss, plt


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
def _(cars_test_features, cars_test_labels, model_1):
    test_results = {}

    test_results['model_1'] = model_1.evaluate(
        cars_test_features[['horsepower','displacement','cylinders']],
        cars_test_labels, verbose=0)
    return (test_results,)


@app.cell
def _(hors_dis_cyl_normalizer, tf):
    model_2 = tf.keras.Sequential([
        hors_dis_cyl_normalizer,
        tf.keras.layers.Dense(8, activation='sigmoid'),
        tf.keras.layers.Dense(3)
    ])

    model_2_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    model_2.compile(
        optimizer=model_2_optimizer, 
        loss='binary_crossentropy', 
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    model_2.summary()
    return model_2, model_2_optimizer


@app.cell
def _(Timer, cars_train_features, cars_train_labels, model_2):
    with Timer():
        model_2_history_1 = model_2.fit(
            cars_train_features[['horsepower','displacement','cylinders']],
            cars_train_labels,
            epochs=300,
            verbose=0,
            validation_split=0.2
        )
    return (model_2_history_1,)


@app.cell
def _(model_2_history_1, plot_loss):
    plot_loss(model_2_history_1)
    return


@app.cell
def _(model_2_history_1, show_history):
    show_history(model_2_history_1)
    return


@app.cell
def _(cars_test_features, cars_test_labels, model_2, test_results):
    test_results['model_2'] = model_2.evaluate(
        cars_test_features[['horsepower','displacement','cylinders']],
        cars_test_labels, verbose=0)
    return


@app.cell
def _(hors_dis_cyl_normalizer, tf):
    model_3 = tf.keras.Sequential([
        hors_dis_cyl_normalizer,
        tf.keras.layers.Dense(64, activation='sigmoid'),
        tf.keras.layers.Dense(64, activation='sigmoid'),
        tf.keras.layers.Dense(3)
    ])

    model_3_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    model_3.compile(
        optimizer=model_3_optimizer, 
        loss='binary_crossentropy', 
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    model_3.summary()
    return model_3, model_3_optimizer


@app.cell
def _(Timer, cars_train_features, cars_train_labels, model_3):
    with Timer():
        model_3_history_1 = model_3.fit(
            cars_train_features[['horsepower','displacement','cylinders']],
            cars_train_labels,
            epochs=100,
            verbose=0,
            validation_split=0.2
        )
    return (model_3_history_1,)


@app.cell
def _(model_3_history_1, plot_loss):
    plot_loss(model_3_history_1)
    return


@app.cell
def _(model_3_history_1, show_history):
    show_history(model_3_history_1)
    return


@app.cell
def _(cars_test_features, cars_test_labels, model_3, test_results):
    test_results['model_3'] = model_3.evaluate(
        cars_test_features[['horsepower','displacement','cylinders']],
        cars_test_labels, verbose=0)
    return


@app.cell
def _(cars_train_features, np, tf):
    more_features_array = np.array(cars_train_features)

    more_features_normalizer = tf.keras.layers.Normalization(axis=-1)

    more_features_normalizer.adapt(more_features_array)
    return more_features_array, more_features_normalizer


@app.cell
def _(more_features_normalizer, tf):
    model_4 = tf.keras.Sequential([
        more_features_normalizer,
        tf.keras.layers.Dense(64, activation='sigmoid'),
        tf.keras.layers.Dense(64, activation='sigmoid'),
        tf.keras.layers.Dense(3)
    ])

    model_4_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    model_4.compile(
        optimizer=model_4_optimizer, 
        loss='binary_crossentropy', 
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    model_4.summary()
    return model_4, model_4_optimizer


@app.cell
def _(Timer, cars_train_features, cars_train_labels, model_4):
    with Timer():
        model_4_history_1 = model_4.fit(
            cars_train_features,
            cars_train_labels,
            epochs=100,
            verbose=0,
            validation_split=0.2
        )
    return (model_4_history_1,)


@app.cell
def _(model_3_history_1, show_history):
    show_history(model_3_history_1)
    return


@app.cell
def _(model_4_history_1, plot_loss):
    plot_loss(model_4_history_1)
    return


@app.cell
def _(cars_test_features, cars_test_labels, model_4, test_results):
    test_results['model_4'] = model_4.evaluate(
        cars_test_features,
        cars_test_labels, verbose=0)
    return


@app.cell
def _(more_features_normalizer, tf):
    model_5 = tf.keras.Sequential([
        more_features_normalizer,
        tf.keras.layers.Dense(128, activation='sigmoid'),
        tf.keras.layers.Dense(128, activation='sigmoid'),
        tf.keras.layers.Dense(3)
    ])

    model_5_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    model_5.compile(
        optimizer=model_5_optimizer, 
        loss='binary_crossentropy', 
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    model_5.summary()
    return model_5, model_5_optimizer


@app.cell
def _(Timer, cars_train_features, cars_train_labels, model_5):
    with Timer():
        model_5_history_1 = model_5.fit(
            cars_train_features,
            cars_train_labels,
            epochs=100,
            verbose=0,
            validation_split=0.2
        )
    return (model_5_history_1,)


@app.cell
def _(model_5_history_1, plot_loss):
    plot_loss(model_5_history_1)
    return


@app.cell
def _(model_5_history_1, show_history):
    show_history(model_5_history_1)
    return


@app.cell
def _(cars_test_features, cars_test_labels, model_5, test_results):
    test_results['model_5'] = model_5.evaluate(
        cars_test_features,
        cars_test_labels, verbose=0)
    return


@app.cell
def _(test_results):
    test_results
    return


@app.cell
def _(pd, test_results):
    result = pd.DataFrame(test_results).transpose()
    result
    return (result,)


if __name__ == "__main__":
    app.run()
