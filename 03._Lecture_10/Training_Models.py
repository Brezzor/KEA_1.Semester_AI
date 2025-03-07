import marimo

__generated_with = "0.11.12"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""# Training Different Models""")
    return


@app.cell
def _():
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_openml
    return fetch_openml, np, plt, time


@app.cell
def _(fetch_openml):
    mnist = fetch_openml('mnist_784', data_home='~/datasets/mnist', as_frame=False, parser='liac-arff')
    return (mnist,)


@app.cell
def _(mnist):
    mnist.data.shape
    return


@app.cell
def _(mnist, np):
    Z = np.c_[mnist.target, mnist.data]
    Z[:,0]
    return (Z,)


@app.cell
def _(Z, np, plt):
    X = Z[:,1:]
    Y = Z[:,0]

    idx = 1030
    X = np.asarray(X, dtype=int)
    print(Y[idx])
    img = plt.imshow(X[idx].reshape(28,28), cmap='gray_r')
    return X, Y, idx, img


@app.cell
def _(X, Y):
    X_train = X[0:50000]
    Y_train = Y[0:50000]

    X_val = X[50000:60000]
    Y_val = Y[50000:60000]

    X_test = X[60000:70000]
    Y_test = Y[60000:70000]
    return X_test, X_train, X_val, Y_test, Y_train, Y_val


@app.cell
def _(X_train):
    X_train.shape
    return


@app.cell
def _():
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    return (
        AdaBoostClassifier,
        DecisionTreeClassifier,
        GaussianNB,
        GaussianProcessClassifier,
        KNeighborsClassifier,
        MLPClassifier,
        QuadraticDiscriminantAnalysis,
        RBF,
        RandomForestClassifier,
        SVC,
    )


@app.cell
def _():
    classifier_names = [
        "Decision Tree",
        "Random Forest", 
        "Neural Net (75, 75)", 
        "Neural Net (784, 784, 784)", 
        "Naive Bayes"
    ]
    return (classifier_names,)


@app.cell
def _(
    DecisionTreeClassifier,
    GaussianNB,
    MLPClassifier,
    RandomForestClassifier,
):
    classifiers = [
        DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=100),
        MLPClassifier(hidden_layer_sizes=(75, 75)),
        MLPClassifier(hidden_layer_sizes=(784, 784, 784)),
        GaussianNB(),
    ]
    return (classifiers,)


@app.cell
def _(X_test, X_train, Y_test, Y_train, classifier_names, classifiers, time):
    for clf, clf_name in zip(classifiers, classifier_names):
        print(f"** {clf_name}")
        t0 = time.time()
        clf.fit(X_train, Y_train)
        t1 = time.time()
        print(f"\tTraining time:\t\t{t1-t0:3.3f}")
        score_train = clf.score(X_train[0:10000], Y_train[0:10000])
        t2 = time.time()
        print(f"\tPrediction time(train):\t{t2-t1:3.3f}")
        score_test = clf.score(X_test, Y_test)
        t3 = time.time()
        print(f"\tPrediction time(test):\t{t3-t2:3.3f}")
        print(f"\tScore Train: {score_train:.3f}\tScore Test: {score_test:.3f}")
    return clf, clf_name, score_test, score_train, t0, t1, t2, t3


@app.cell
def _(MLPClassifier, X_test, X_train, Y_test, Y_train, time):
    # default alpha=0.0001
    for a in [0.0001, 0.001, 0.01, 0.1, 1]:
        mlp = MLPClassifier(hidden_layer_sizes=(75, 75), alpha=a)
        t0 = time.time()
        mlp.fit(X_train, Y_train)
        t1 = time.time()
        print(mlp.score(X_test, Y_test), t1 - t0)
    return a, mlp, t0, t1


@app.cell
def _(MLPClassifier, X_test, X_train, Y_test, Y_train, time):
    for hl in [(25), (50), (50, 50), (100), (100, 100)]:
        mlp = MLPClassifier(hidden_layer_sizes=hl)
        t0 = time.time()
        mlp.fit(X_train, Y_train)
        t1 = time.time()
        print(mlp.score(X_test, Y_test), t1 - t0)
    return hl, mlp, t0, t1


if __name__ == "__main__":
    app.run()
