import marimo

__generated_with = "0.11.24"
app = marimo.App()


@app.cell
def _():
    import tensorflow as tf
    return (tf,)


@app.cell
def _():
    import tensorflow_datasets as tfds
    return (tfds,)


@app.cell
def _(tfds):
    (ds_test, ds_val, ds_train), ds_info = tfds.load(
        'mnist',
        split=['test', 'train[0%:17%]', 'train[17%:]'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
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
    ds_train_1 = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train_1 = ds_train_1.cache()
    ds_train_1 = ds_train_1.shuffle(len(ds_train_1))
    ds_train_1 = ds_train_1.batch(128)
    ds_train_1 = ds_train_1.prefetch(tf.data.AUTOTUNE)
    return (ds_train_1,)


@app.cell
def _(ds_val, normalize_img, tf):
    ds_val_1 = ds_val.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_val_1 = ds_val_1.cache()
    ds_val_1 = ds_val_1.batch(128)
    ds_val_1 = ds_val_1.prefetch(tf.data.AUTOTUNE)
    return (ds_val_1,)


@app.cell
def _(ds_test, normalize_img, tf):
    ds_test_1 = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test_1 = ds_test_1.cache()
    ds_test_1 = ds_test_1.batch(128)
    ds_test_1 = ds_test_1.prefetch(tf.data.AUTOTUNE)
    return (ds_test_1,)


@app.cell
def _():
    history = []
    return (history,)


@app.cell
def _(tf):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(384, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.summary()
    return (model,)


@app.cell
def _(model, tf):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    return


@app.cell
def _(ds_train_1, ds_val_1, history, model):
    history.append(('Basic DNN', model.fit(ds_train_1, epochs=25, validation_data=ds_val_1)))
    return


@app.cell
def _(ds_test_1, model):
    model.evaluate(ds_test_1)
    return


@app.cell
def _(model):
    del model
    return


@app.cell
def _(tf):
    model_1 = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(384, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.005)), tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.005)), tf.keras.layers.Dense(10)])
    return (model_1,)


@app.cell
def _(model_1, tf):
    model_1.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    model_1.summary()
    return


@app.cell
def _(ds_train_1, ds_val_1, history, model_1):
    history.append(('Basic DNN reg.', model_1.fit(ds_train_1, epochs=10, validation_data=ds_val_1)))
    return


@app.cell
def _(ds_test_1, model_1):
    model_1.evaluate(ds_test_1)
    return


@app.cell
def _(model_1):
    del model_1
    return


@app.cell
def _(tf):
    model_2 = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(384, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)), tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)), tf.keras.layers.Dense(10)])
    return (model_2,)


@app.cell
def _(model_2, tf):
    model_2.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return


@app.cell
def _(ds_train_1, ds_val_1, history, model_2):
    history.append(('DNN reg. long training.', model_2.fit(ds_train_1, epochs=25, validation_data=ds_val_1)))
    return


@app.cell
def _(ds_test_1, model_2):
    model_2.evaluate(ds_test_1)
    return


@app.cell
def _(model_2):
    del model_2
    return


@app.cell
def _(tf):
    model_3 = tf.keras.Sequential([tf.keras.Input(shape=(28, 28, 1)), tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'), tf.keras.layers.MaxPooling2D(pool_size=(2, 2)), tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'), tf.keras.layers.MaxPooling2D(pool_size=(2, 2)), tf.keras.layers.Flatten(), tf.keras.layers.Dropout(0.5), tf.keras.layers.Dense(10, activation='linear')])
    model_3.summary()
    return (model_3,)


@app.cell
def _(model_3, tf):
    model_3.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return


@app.cell
def _(ds_train_1, ds_val_1, history, model_3):
    history.append(('CNN', model_3.fit(ds_train_1, epochs=25, validation_data=ds_val_1)))
    return


@app.cell
def _(ds_test_1, model_3):
    model_3.evaluate(ds_test_1)
    return


@app.cell
def _(history):
    history
    return


@app.cell
def _(history):
    hist_elements = list(history[0][1].history.keys())
    hist_elements
    return (hist_elements,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    return (plt,)


@app.cell
def _(hist_elements, history, plt):
    for i in hist_elements:
        print(i)
        for j in history:
            plt.plot(range(len(j[1].history[i])), j[1].history[i], label=j[0])
            plt.legend()
        plt.show()
    return i, j


if __name__ == "__main__":
    app.run()
