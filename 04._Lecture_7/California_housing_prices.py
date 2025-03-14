import marimo

__generated_with = "0.11.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import statistics
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from pandas.plotting import scatter_matrix
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler, StandardScaler, FunctionTransformer
    from sklearn.metrics.pairwise import rbf_kernel
    from sklearn.linear_model import LinearRegression
    from sklearn.compose import TransformedTargetRegressor
    return (
        FunctionTransformer,
        LinearRegression,
        MinMaxScaler,
        OneHotEncoder,
        OrdinalEncoder,
        SimpleImputer,
        StandardScaler,
        TransformedTargetRegressor,
        mo,
        np,
        pd,
        plt,
        rbf_kernel,
        scatter_matrix,
        statistics,
        train_test_split,
    )


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
def _(strat_test_set, strat_train_set):
    for set in (strat_train_set, strat_test_set):
        set.drop("income_cat", axis=1, inplace=True)
    return (set,)


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


@app.cell
def _(strat_train_set):
    housing_values = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()
    housing_values.info()
    return housing_labels, housing_values


@app.cell
def _(housing_values):
    median = housing_values["total_bedrooms"].median()
    housing_values["total_bedrooms"] = housing_values["total_bedrooms"].fillna(median)
    housing_values.info()
    return (median,)


@app.cell
def _(SimpleImputer, housing_values, np):
    imputer = SimpleImputer(strategy="median")
    housing_num = housing_values.select_dtypes(include=[np.number])
    imputer.fit(housing_num)
    return housing_num, imputer


@app.cell
def _(imputer):
    imputer.statistics_
    return


@app.cell
def _(housing_num):
    housing_num.median().values
    return


@app.cell
def _(housing_num, imputer):
    X = imputer.transform(housing_num)
    X
    return (X,)


@app.cell
def _(X, housing_num, pd):
    housing_tr = pd.DataFrame(X, columns=housing_num.columns,index=housing_num.index)
    housing_tr.head()
    return (housing_tr,)


@app.cell
def _(housing_values):
    housing_cat = housing_values[["ocean_proximity"]]
    housing_cat.head()
    return (housing_cat,)


@app.cell
def _(OrdinalEncoder, housing_cat):
    ordinal_encoder = OrdinalEncoder()
    housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
    housing_cat_encoded[:8]
    return housing_cat_encoded, ordinal_encoder


@app.cell
def _(ordinal_encoder):
    ordinal_encoder.categories_
    return


@app.cell
def _(OneHotEncoder, housing_cat):
    cat_encoder = OneHotEncoder()
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
    housing_cat_1hot
    return cat_encoder, housing_cat_1hot


@app.cell
def _(housing_cat_1hot):
    housing_cat_1hot.toarray()
    return


@app.cell
def _(cat_encoder):
    cat_encoder.categories_
    return


@app.cell
def _(pd):
    df_test = pd.DataFrame({"ocean_proximity": ["INLAND", "NEAR BAY"]})
    pd.get_dummies(df_test)
    return (df_test,)


@app.cell
def _(cat_encoder, df_test):
    cat_encoder.transform(df_test).toarray()
    return


@app.cell
def _(pd):
    df_test_unknown = pd.DataFrame({"ocean_proximity": ["<2H OCEAN", "ISLAND"]})
    pd.get_dummies(df_test_unknown)
    return (df_test_unknown,)


@app.cell
def _(cat_encoder, df_test_unknown):
    cat_encoder.handle_unknown = "ignore"
    cat_encoder.transform(df_test_unknown).toarray()
    return


@app.cell
def _(cat_encoder):
    cat_encoder.feature_names_in_
    return


@app.cell
def _(cat_encoder):
    cat_encoder.get_feature_names_out()
    return


@app.cell
def _(cat_encoder, df_test_unknown, pd):
    try:
        df_output = pd.DataFrame(cat_encoder.transform(df_test_unknown), columns=cat_encoder.get_feature_names_out(), index=df_test_unknown.index)
    except ValueError as E:
        print(E)
    return (df_output,)


@app.cell
def _(MinMaxScaler, housing_num):
    min_max_scaler = MinMaxScaler(feature_range=(-1,1))
    housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)
    housing_num_min_max_scaled[:5]
    return housing_num_min_max_scaled, min_max_scaler


@app.cell
def _(StandardScaler, housing_num):
    std_scaler = StandardScaler()
    housing_num_std_scaled = std_scaler.fit_transform(housing_num)
    housing_num_std_scaled[:5]
    return housing_num_std_scaled, std_scaler


@app.cell
def _(housing_values):
    housing_median_age = housing_values[["housing_median_age"]]
    housing_median_age.head()
    return (housing_median_age,)


@app.cell
def _(housing_median_age, rbf_kernel):
    age_simil_35 = rbf_kernel(housing_median_age, [[35]], gamma=0.1)
    age_simil_35
    return (age_simil_35,)


@app.cell
def _(LinearRegression, StandardScaler, housing_labels, housing_values):
    target_scaler = StandardScaler()
    scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())

    linear_model1 = LinearRegression()
    linear_model1.fit(housing_values[["median_income"]], scaled_labels)
    some_new_data = housing_values[["median_income"]].iloc[:5]

    scaled_predictions = linear_model1.predict(some_new_data)
    predictions1 = target_scaler.inverse_transform(scaled_predictions)
    predictions1
    return (
        linear_model1,
        predictions1,
        scaled_labels,
        scaled_predictions,
        some_new_data,
        target_scaler,
    )


@app.cell
def _(
    LinearRegression,
    StandardScaler,
    TransformedTargetRegressor,
    housing_labels,
    housing_values,
    some_new_data,
):
    treg_model1 = TransformedTargetRegressor(LinearRegression(), transformer=StandardScaler())
    treg_model1.fit(housing_values[["median_income"]], housing_labels)
    predictions2 = treg_model1.predict(some_new_data)
    predictions2
    return predictions2, treg_model1


@app.cell
def _(FunctionTransformer, housing_values, np):
    log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
    log_pop = log_transformer.transform(housing_values[["population"]])
    log_pop.head()
    return log_pop, log_transformer


@app.cell
def _(FunctionTransformer, housing_values, rbf_kernel):
    rbf_transformer = FunctionTransformer(rbf_kernel, kw_args=dict(Y=[[35.]], gamma=0.1))
    age_simil_35_2 = rbf_transformer.transform(housing_values[["housing_median_age"]])
    age_simil_35_2
    return age_simil_35_2, rbf_transformer


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
