import marimo

__generated_with = "0.11.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from pandas.plotting import scatter_matrix
    return mo, np, pd, plt, scatter_matrix, train_test_split


@app.cell
def _(pd):
    housing_df = pd.read_csv("Datasets\Housing.csv")
    housing_df.head()
    return (housing_df,)


@app.cell
def _(housing_df):
    housing_df.info()
    return


@app.cell
def _(housing_df):
    housing_df["ocean_proximity"].value_counts()
    return


@app.cell
def _(housing_df):
    housing_df.describe()
    return


@app.cell
def _(housing_df, plt):
    housing_df.hist(bins=50, figsize=(12,8))
    plt.show()
    return


@app.cell
def _(housing_df, train_test_split):
    train_set, test_set = train_test_split(housing_df, test_size=0.2, random_state=42)
    print(f"Training set: {len(train_set)}")
    print(f"Test set: {len(test_set)}")
    return test_set, train_set


@app.cell
def _(housing_df, np, pd, plt):
    housing_df["income_cat"] = pd.cut(housing_df["median_income"],bins=[0.,1.5,3.0,4.5,6, np.inf], labels=[1,2,3,4,5])
    housing_df["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
    plt.xlabel("Income category")
    plt.ylabel("Number of districts")
    plt.show()
    return


@app.cell
def _(housing_df, train_test_split):
    strat_train_set, strat_test_set = train_test_split(housing_df, test_size=0.2, stratify=housing_df["income_cat"], random_state=42)
    strat_test_set["income_cat"].value_counts() / len(strat_test_set)
    return strat_test_set, strat_train_set


@app.cell
def _(plt, strat_train_set):
    housing = strat_train_set.copy()
    housing.plot(kind="scatter", x="longitude", y="latitude", grid=True)
    plt.show()
    return (housing,)


@app.cell
def _(housing, plt):
    housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)
    plt.show()
    return


@app.cell
def _(housing, plt):
    housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, s=housing["population"] / 100, label="population", c="median_house_value", cmap="jet", colorbar=True, legend=True, sharex=False, figsize=(10,7))
    plt.show()
    return


@app.cell
def _(housing):
    corr_matrix1 = housing[["median_house_value","median_income","total_rooms","housing_median_age","households","total_bedrooms","population","longitude","latitude"]].corr()
    corr_matrix1["median_house_value"].sort_values(ascending=False)
    return (corr_matrix1,)


@app.cell
def _(housing, plt, scatter_matrix):
    attributes = ["median_house_value","median_income","total_rooms","housing_median_age"]
    scatter_matrix(housing[attributes], figsize=(12,8))
    plt.show()
    return (attributes,)


@app.cell
def _(housing, plt):
    housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1, grid=True)
    plt.show()
    return


@app.cell
def _(housing):
    housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["people_per_house"] = housing["population"] / housing["households"]

    corr_matrix2 = housing[["median_house_value","median_income","total_rooms","housing_median_age","households","total_bedrooms","population","longitude","latitude","rooms_per_house","bedrooms_ratio","people_per_house"]].corr()
    corr_matrix2["median_house_value"].sort_values(ascending=False)
    return (corr_matrix2,)


if __name__ == "__main__":
    app.run()
