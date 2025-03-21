import marimo

__generated_with = "0.11.24"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""# Tensorflow MNIST Classification""")
    return


@app.cell
def _():
    import marimo as mo
    import tensorflow as tf
    import tensorflow_datasets as tfds
    return mo, tf, tfds


@app.cell
def _(tfds):
    (ds_test, ds_val, ds_train), ds_info = tfds.load(
        'mnist',
        split=['test', 'train[0%:17%]', 'train[17%:]'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )
    return ds_info, ds_test, ds_train, ds_val


@app.cell
def _(ds_test, ds_train, ds_val):
    len(ds_train), len(ds_val), len(ds_test)
    return


@app.cell
def _(tf):
    def normalize_img(image, label):
        return tf.cast(image, tf.float32) / 255., label
    return (normalize_img,)


@app.cell
def _(ds_train, normalize_img, tf):
    training_set = ds_train.map(
        normalize_img,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    training_set = ds_train.cache()
    training_set = ds_train.shuffle(len(ds_train))
    training_set = ds_train.batch(128)
    training_set = ds_train.prefetch(tf.data.AUTOTUNE)
    return (training_set,)


@app.cell
def _(ds_val, normalize_img, tf):
    validation_set = ds_val.map(
        normalize_img, 
        num_parallel_calls=tf.data.AUTOTUNE
    )

    validation_set = ds_val.cache()
    validation_set = ds_val.batch(128)
    validation_set = ds_val.prefetch(tf.data.AUTOTUNE)
    return (validation_set,)


@app.cell
def _(ds_test, normalize_img, tf):
    test_set = ds_test.map(
        normalize_img, 
        num_parallel_calls=tf.data.AUTOTUNE
    )

    test_set = ds_test.cache()
    test_set = ds_test.batch(128)
    test_set = ds_test.prefetch(tf.data.AUTOTUNE)
    return (test_set,)


@app.cell
def _():
    history = []
    return (history,)


@app.cell
def _(mo):
    mo.md(r"""# Using a basic network""")
    return


@app.cell
def _(tf):
    model_1 = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(384, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model_1.summary()
    return (model_1,)


@app.cell
def _(model_1, tf):
    model_1.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    return


@app.cell
def _(history, model_1, training_set, validation_set):
    history.append(('Basic DNN', model_1.fit(
        training_set,
        epochs=10,
        validation_data=validation_set,
    )))
    return


if __name__ == "__main__":
    app.run()
