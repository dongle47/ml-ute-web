import streamlit as st

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

st.markdown("# Cali Housing Demo")

st.sidebar.header("Cali Housing Exercises")

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

def display_scores(scores):
    st.write("Mean: %.2f" % (scores.mean()))
    st.write("Standard deviation: %.2f" % (scores.std()))

def DecisionTreeRegression():
    housing = pd.read_csv("housing.csv")

    housing["income_cat"] = pd.cut(housing["median_income"],
                                bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                labels=[1, 2, 3, 4, 5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    housing_num = housing.drop("ocean_proximity", axis=1)

    num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('attribs_adder', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler()),
        ])

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ])

    housing_prepared = full_pipeline.fit_transform(housing)

    # Training
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(housing_prepared, housing_labels)

    # Prediction
    some_data = housing.iloc[:5]
    some_labels = housing_labels.iloc[:5]
    some_data_prepared = full_pipeline.transform(some_data)
    # Prediction 5 samples 
    st.write("Predictions:", tree_reg.predict(some_data_prepared))
    st.write("Labels:", list(some_labels))
    st.write('\n')

    # T??nh sai s??? b??nh ph????ng trung b??nh tr??n t???p d??? li???u hu???n luy???n
    housing_predictions = tree_reg.predict(housing_prepared)
    mse_train = mean_squared_error(housing_labels, housing_predictions)
    rmse_train = np.sqrt(mse_train)
    st.write('Sai s??? b??nh ph????ng trung b??nh - train:')
    st.write('%.2f' % rmse_train)

    # T??nh sai s??? b??nh ph????ng trung b??nh tr??n t???p d??? li???u ki???m ?????nh ch??o (cross-validation) 
    scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

    st.write('Sai s??? b??nh ph????ng trung b??nh - cross-validation:')
    rmse_cross_validation = np.sqrt(-scores)
    display_scores(rmse_cross_validation)

    # T??nh sai s??? b??nh ph????ng trung b??nh tr??n t???p d??? li???u ki???m tra (test)
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()
    X_test_prepared = full_pipeline.transform(X_test)
    y_predictions = tree_reg.predict(X_test_prepared)

    mse_test = mean_squared_error(y_test, y_predictions)
    rmse_test = np.sqrt(mse_test)
    st.write('Sai s??? b??nh ph????ng trung b??nh - test:')
    st.write('%.2f' % rmse_test)

def LinearRegressionUseModel():
    # housing = pd.read_csv("E:/UTE/MachineLearning/Thu5/End_to_End_Project/CaliHousing/housing.csv")
    housing = pd.read_csv("housing.csv")

# Them column income_cat dung de chia data
    housing["income_cat"] = pd.cut(housing["median_income"],
                                bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                labels=[1, 2, 3, 4, 5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    # Chia xong thi delete column income_cat
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    housing_num = housing.drop("ocean_proximity", axis=1)

    num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('attribs_adder', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler()),
        ])

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ])

    housing_prepared = full_pipeline.fit_transform(housing)

    # Load model lin_reg to use
    lin_reg = LinearRegression()
    lin_reg = joblib.load("model_lin_reg.pkl")


    # Prediction
    some_data = housing.iloc[:5]
    some_labels = housing_labels.iloc[:5]
    some_data_prepared = full_pipeline.transform(some_data)
    # Prediction 5 samples 
    st.write("Predictions:", lin_reg.predict(some_data_prepared))
    st.write("Labels:", list(some_labels))
    st.write('\n')

    # T??nh sai s??? b??nh ph????ng trung b??nh tr??n t???p d??? li???u hu???n luy???n
    housing_predictions = lin_reg.predict(housing_prepared)
    mse_train = mean_squared_error(housing_labels, housing_predictions)
    rmse_train = np.sqrt(mse_train)
    st.write('Sai s??? b??nh ph????ng trung b??nh - train:')
    st.write('%.2f' % rmse_train)

    # T??nh sai s??? b??nh ph????ng trung b??nh tr??n t???p d??? li???u ki???m ?????nh ch??o (cross-validation) 
    scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

    st.write('Sai s??? b??nh ph????ng trung b??nh - cross-validation:')
    rmse_cross_validation = np.sqrt(-scores)
    display_scores(rmse_cross_validation)

    # T??nh sai s??? b??nh ph????ng trung b??nh tr??n t???p d??? li???u ki???m tra (test)
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()
    X_test_prepared = full_pipeline.transform(X_test)
    y_predictions = lin_reg.predict(X_test_prepared)

    mse_test = mean_squared_error(y_test, y_predictions)
    rmse_test = np.sqrt(mse_test)
    st.write('Sai s??? b??nh ph????ng trung b??nh - test:')
    st.write('%.2f' % rmse_test)

def LinearRegressionEx():
    housing = pd.read_csv("housing.csv")

    housing["income_cat"] = pd.cut(housing["median_income"],
                                bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                labels=[1, 2, 3, 4, 5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    # Chia xong thi delete column income_cat
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    housing_num = housing.drop("ocean_proximity", axis=1)

    num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('attribs_adder', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler()),
        ])

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ])

    housing_prepared = full_pipeline.fit_transform(housing)

    # Training
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)

    # Save model lin_reg 
    joblib.dump(lin_reg, "model_lin_reg.pkl")

    # Prediction
    some_data = housing.iloc[:5]
    some_labels = housing_labels.iloc[:5]
    some_data_prepared = full_pipeline.transform(some_data)
    # Prediction 5 samples 
    st.write("Predictions:", lin_reg.predict(some_data_prepared))
    st.write("Labels:", list(some_labels))
    st.write('\n')

    # T??nh sai s??? b??nh ph????ng trung b??nh tr??n t???p d??? li???u hu???n luy???n
    housing_predictions = lin_reg.predict(housing_prepared)
    mse_train = mean_squared_error(housing_labels, housing_predictions)
    rmse_train = np.sqrt(mse_train)
    st.write('Sai s??? b??nh ph????ng trung b??nh - train:')
    st.write('%.2f' % rmse_train)

    # T??nh sai s??? b??nh ph????ng trung b??nh tr??n t???p d??? li???u ki???m ?????nh ch??o (cross-validation) 
    scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

    st.write('Sai s??? b??nh ph????ng trung b??nh - cross-validation:')
    rmse_cross_validation = np.sqrt(-scores)
    display_scores(rmse_cross_validation)

    # T??nh sai s??? b??nh ph????ng trung b??nh tr??n t???p d??? li???u ki???m tra (test)
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()
    X_test_prepared = full_pipeline.transform(X_test)
    y_predictions = lin_reg.predict(X_test_prepared)

    mse_test = mean_squared_error(y_test, y_predictions)
    rmse_test = np.sqrt(mse_test)
    st.write('Sai s??? b??nh ph????ng trung b??nh - test:')
    st.write('%.2f' % rmse_test)

def RandomForestRegression():
    housing = pd.read_csv("housing.csv")
# Them column income_cat dung de chia data
    housing["income_cat"] = pd.cut(housing["median_income"],
                                bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                labels=[1, 2, 3, 4, 5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    # Chia xong thi delete column income_cat
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    housing_num = housing.drop("ocean_proximity", axis=1)

    num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('attribs_adder', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler()),
        ])

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ])

    housing_prepared = full_pipeline.fit_transform(housing)

    # Training
    forest_reg = RandomForestRegressor()
    forest_reg.fit(housing_prepared, housing_labels)


    # Prediction
    some_data = housing.iloc[:5]
    some_labels = housing_labels.iloc[:5]
    some_data_prepared = full_pipeline.transform(some_data)
    # Prediction 5 samples 
    st.write("Predictions:", forest_reg.predict(some_data_prepared))
    st.write("Labels:", list(some_labels))
    st.write('\n')

    # T??nh sai s??? b??nh ph????ng trung b??nh tr??n t???p d??? li???u hu???n luy???n
    housing_predictions = forest_reg.predict(housing_prepared)
    mse_train = mean_squared_error(housing_labels, housing_predictions)
    rmse_train = np.sqrt(mse_train)
    st.write('Sai s??? b??nh ph????ng trung b??nh - train:')
    st.write('%.2f' % rmse_train)

    # T??nh sai s??? b??nh ph????ng trung b??nh tr??n t???p d??? li???u ki???m ?????nh ch??o (cross-validation) 
    scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

    st.write('Sai s??? b??nh ph????ng trung b??nh - cross-validation:')
    rmse_cross_validation = np.sqrt(-scores)
    display_scores(rmse_cross_validation)

    # T??nh sai s??? b??nh ph????ng trung b??nh tr??n t???p d??? li???u ki???m tra (test)
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()
    X_test_prepared = full_pipeline.transform(X_test)
    y_predictions = forest_reg.predict(X_test_prepared)

    mse_test = mean_squared_error(y_test, y_predictions)
    rmse_test = np.sqrt(mse_test)
    st.write('Sai s??? b??nh ph????ng trung b??nh - test:')
    st.write('%.2f' % rmse_test)

# def RandomForestRegressionRandomSearchCV():
#     housing = pd.read_csv("housing.csv")
    
#     housing["income_cat"] = pd.cut(housing["median_income"],
#                                 bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
#                                 labels=[1, 2, 3, 4, 5])

#     split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
#     for train_index, test_index in split.split(housing, housing["income_cat"]):
#         strat_train_set = housing.loc[train_index]
#         strat_test_set = housing.loc[test_index]

#     # Chia xong thi delete column income_cat
#     for set_ in (strat_train_set, strat_test_set):
#         set_.drop("income_cat", axis=1, inplace=True)

#     housing = strat_train_set.drop("median_house_value", axis=1)
#     housing_labels = strat_train_set["median_house_value"].copy()

#     housing_num = housing.drop("ocean_proximity", axis=1)

#     num_pipeline = Pipeline([
#             ('imputer', SimpleImputer(strategy="median")),
#             ('attribs_adder', CombinedAttributesAdder()),
#             ('std_scaler', StandardScaler()),
#         ])

#     num_attribs = list(housing_num)
#     cat_attribs = ["ocean_proximity"]
#     full_pipeline = ColumnTransformer([
#             ("num", num_pipeline, num_attribs),
#             ("cat", OneHotEncoder(), cat_attribs),
#         ])

#     housing_prepared = full_pipeline.fit_transform(housing)

#     param_distribs = {
#             'n_estimators': randint(low=1, high=200),
#             'max_features': randint(low=1, high=8),
#         }

#     # Training
#     forest_reg = RandomForestRegressor(random_state=42)
#     rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
#                                     n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
#     rnd_search.fit(housing_prepared, housing_labels)

#     final_model = rnd_search.best_estimator_
#     joblib.dump(final_model, "forest_reg_rand_search.pkl")


#     # Prediction
#     some_data = housing.iloc[:5]
#     some_labels = housing_labels.iloc[:5]
#     some_data_prepared = full_pipeline.transform(some_data)
#     # Prediction 5 samples 
#     st.write("Predictions:", final_model.predict(some_data_prepared))
#     st.write("Labels:", list(some_labels))
#     st.write('\n')

#     # T??nh sai s??? b??nh ph????ng trung b??nh tr??n t???p d??? li???u hu???n luy???n
#     housing_predictions = final_model.predict(housing_prepared)
#     mse_train = mean_squared_error(housing_labels, housing_predictions)
#     rmse_train = np.sqrt(mse_train)
#     st.write('Sai s??? b??nh ph????ng trung b??nh - train:')
#     st.write('%.2f' % rmse_train)

#     # T??nh sai s??? b??nh ph????ng trung b??nh tr??n t???p d??? li???u ki???m ?????nh ch??o (cross-validation) 
#     scores = cross_val_score(final_model, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

#     st.write('Sai s??? b??nh ph????ng trung b??nh - cross-validation:')
#     rmse_cross_validation = np.sqrt(-scores)
#     display_scores(rmse_cross_validation)

#     # T??nh sai s??? b??nh ph????ng trung b??nh tr??n t???p d??? li???u ki???m tra (test)
#     X_test = strat_test_set.drop("median_house_value", axis=1)
#     y_test = strat_test_set["median_house_value"].copy()
#     X_test_prepared = full_pipeline.transform(X_test)
#     y_predictions = final_model.predict(X_test_prepared)

#     mse_test = mean_squared_error(y_test, y_predictions)
#     rmse_test = np.sqrt(mse_test)
#     st.write('Sai s??? b??nh ph????ng trung b??nh - test:')
#     st.write('%.2f' % rmse_test)

# def RandomForestRegressionRandomSearchCVUseModel():
#     housing = pd.read_csv("housing.csv")

# # Them column income_cat dung de chia data
#     housing["income_cat"] = pd.cut(housing["median_income"],
#                                 bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
#                                 labels=[1, 2, 3, 4, 5])

#     split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
#     for train_index, test_index in split.split(housing, housing["income_cat"]):
#         strat_train_set = housing.loc[train_index]
#         strat_test_set = housing.loc[test_index]

#     # Chia xong thi delete column income_cat
#     for set_ in (strat_train_set, strat_test_set):
#         set_.drop("income_cat", axis=1, inplace=True)

#     housing = strat_train_set.drop("median_house_value", axis=1)
#     housing_labels = strat_train_set["median_house_value"].copy()

#     housing_num = housing.drop("ocean_proximity", axis=1)

#     num_pipeline = Pipeline([
#             ('imputer', SimpleImputer(strategy="median")),
#             ('attribs_adder', CombinedAttributesAdder()),
#             ('std_scaler', StandardScaler()),
#         ])

#     num_attribs = list(housing_num)
#     cat_attribs = ["ocean_proximity"]
#     full_pipeline = ColumnTransformer([
#             ("num", num_pipeline, num_attribs),
#             ("cat", OneHotEncoder(), cat_attribs),
#         ])

#     housing_prepared = full_pipeline.fit_transform(housing)

#     param_distribs = {
#             'n_estimators': randint(low=1, high=200),
#             'max_features': randint(low=1, high=8),
#         }

#     # Training
#     forest_reg = RandomForestRegressor(random_state=42)
#     rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
#                                     n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
#     rnd_search.fit(housing_prepared, housing_labels)

#     final_model = rnd_search.best_estimator_
#     joblib.dump(final_model, "forest_reg_rand_search.pkl")

#     # Prediction
#     some_data = housing.iloc[:5]
#     some_labels = housing_labels.iloc[:5]
#     some_data_prepared = full_pipeline.transform(some_data)
#     # Prediction 5 samples 
#     st.write("Predictions:", final_model.predict(some_data_prepared))
#     st.write("Labels:", list(some_labels))
#     st.write('\n')

#     # T??nh sai s??? b??nh ph????ng trung b??nh tr??n t???p d??? li???u hu???n luy???n
#     housing_predictions = final_model.predict(housing_prepared)
#     mse_train = mean_squared_error(housing_labels, housing_predictions)
#     rmse_train = np.sqrt(mse_train)
#     st.write('Sai s??? b??nh ph????ng trung b??nh - train:')
#     st.write('%.2f' % rmse_train)

#     # T??nh sai s??? b??nh ph????ng trung b??nh tr??n t???p d??? li???u ki???m ?????nh ch??o (cross-validation) 
#     scores = cross_val_score(final_model, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

#     st.write('Sai s??? b??nh ph????ng trung b??nh - cross-validation:')
#     rmse_cross_validation = np.sqrt(-scores)
#     display_scores(rmse_cross_validation)

#     # T??nh sai s??? b??nh ph????ng trung b??nh tr??n t???p d??? li???u ki???m tra (test)
#     X_test = strat_test_set.drop("median_house_value", axis=1)
#     y_test = strat_test_set["median_house_value"].copy()
#     X_test_prepared = full_pipeline.transform(X_test)
#     y_predictions = final_model.predict(X_test_prepared)

#     mse_test = mean_squared_error(y_test, y_predictions)
#     rmse_test = np.sqrt(mse_test)
#     st.write('Sai s??? b??nh ph????ng trung b??nh - test:')
#     st.write('%.2f' % rmse_test)

# def RandomForestRegressionGridSearchCV():
#     # housing = pd.read_csv("E:/UTE/MachineLearning/Thu5/End_to_End_Project/CaliHousing/housing.csv")
#     housing = pd.read_csv("housing.csv")

#     # Them column income_cat dung de chia data
#     housing["income_cat"] = pd.cut(housing["median_income"],
#                                 bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
#                                 labels=[1, 2, 3, 4, 5])

#     split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
#     for train_index, test_index in split.split(housing, housing["income_cat"]):
#         strat_train_set = housing.loc[train_index]
#         strat_test_set = housing.loc[test_index]

#     # Chia xong thi delete column income_cat
#     for set_ in (strat_train_set, strat_test_set):
#         set_.drop("income_cat", axis=1, inplace=True)

#     housing = strat_train_set.drop("median_house_value", axis=1)
#     housing_labels = strat_train_set["median_house_value"].copy()

#     housing_num = housing.drop("ocean_proximity", axis=1)

#     num_pipeline = Pipeline([
#             ('imputer', SimpleImputer(strategy="median")),
#             ('attribs_adder', CombinedAttributesAdder()),
#             ('std_scaler', StandardScaler()),
#         ])

#     num_attribs = list(housing_num)
#     cat_attribs = ["ocean_proximity"]
#     full_pipeline = ColumnTransformer([
#             ("num", num_pipeline, num_attribs),
#             ("cat", OneHotEncoder(), cat_attribs),
#         ])

#     housing_prepared = full_pipeline.fit_transform(housing)

#     param_distribs = {
#             'n_estimators': randint(low=1, high=200),
#             'max_features': randint(low=1, high=8),
#         }

#     # Training
#     forest_reg = RandomForestRegressor(random_state=42)
#     rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
#                                     n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
#     rnd_search.fit(housing_prepared, housing_labels)

#     final_model = rnd_search.best_estimator_
#     joblib.dump(final_model, "forest_reg_rand_search.pkl")


#     # Prediction
#     some_data = housing.iloc[:5]
#     some_labels = housing_labels.iloc[:5]
#     some_data_prepared = full_pipeline.transform(some_data)
#     # Prediction 5 samples 
#     st.write("Predictions:", final_model.predict(some_data_prepared))
#     st.write("Labels:", list(some_labels))
#     st.write('\n')

#     # T??nh sai s??? b??nh ph????ng trung b??nh tr??n t???p d??? li???u hu???n luy???n
#     housing_predictions = final_model.predict(housing_prepared)
#     mse_train = mean_squared_error(housing_labels, housing_predictions)
#     rmse_train = np.sqrt(mse_train)
#     st.write('Sai s??? b??nh ph????ng trung b??nh - train:')
#     st.write('%.2f' % rmse_train)

#     # T??nh sai s??? b??nh ph????ng trung b??nh tr??n t???p d??? li???u ki???m ?????nh ch??o (cross-validation) 
#     scores = cross_val_score(final_model, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

#     st.write('Sai s??? b??nh ph????ng trung b??nh - cross-validation:')
#     rmse_cross_validation = np.sqrt(-scores)
#     display_scores(rmse_cross_validation)

#     # T??nh sai s??? b??nh ph????ng trung b??nh tr??n t???p d??? li???u ki???m tra (test)
#     X_test = strat_test_set.drop("median_house_value", axis=1)
#     y_test = strat_test_set["median_house_value"].copy()
#     X_test_prepared = full_pipeline.transform(X_test)
#     y_predictions = final_model.predict(X_test_prepared)

#     mse_test = mean_squared_error(y_test, y_predictions)
#     rmse_test = np.sqrt(mse_test)
#     st.write('Sai s??? b??nh ph????ng trung b??nh - test:')
#     st.write('%.2f' % rmse_test)


option = st.sidebar.selectbox('L???a ch???n b??i t???p',
    ('Decision Tree Regression', 'Linear Regression', 'Linear Regression UseModel', 
    'Random Forest Regression', 
    # 'Random Forest Regression Random Search CV', 
    # 'Random Forest Regression Random Search CV UseModel',
    # 'Random Forest Regression Grid Search CV'
    ))

if(option == 'Decision Tree Regression'):
    DecisionTreeRegression()
if(option == 'Linear Regression'):
    LinearRegressionEx()
if(option == 'Linear Regression UseModel'):
    LinearRegressionUseModel()
if(option == 'Random Forest Regression'):
    RandomForestRegression()
# if(option == 'Random Forest Regression Random Search CV'):
#     RandomForestRegressionRandomSearchCV()
# if(option == 'Random Forest Regression Random Search CV UseModel'):
#     RandomForestRegressionRandomSearchCVUseModel()
# if(option == 'Random Forest Regression Grid Search CV'):
#     RandomForestRegressionGridSearchCV()

