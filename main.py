import pandas as pd
# import ydata_profiling
# import sweetviz
# import autoviz
import numpy as np
from sklearn.linear_model import LinearRegression, ElasticNet, SGDRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVR
import dtale
import os
import xgboost

def eda(csvfile):
    if not os.path.exists("reports"):
        os.mkdir("reports")
    # profile = ydata_profiling.ProfileReport(pd.read_csv(csvfile), explorative=True)
    # profile.to_file("reports/profile.html")
    # AV = autoviz.AutoViz_Class()
    # auto = AV.AutoViz(filename=csvfile, chart_format='html', save_plot_dir="reports")
    # sv = sweetviz.analyze(pd.read_csv(csvfile))
    # sv.show_html("reports/sweetviz.html")
    dtale.show(pd.read_csv(csvfile), subprocess=False)
    return 0


def preprocess(csvfile):
    df = pd.read_csv(csvfile)
    df = df.drop("id", axis=1)
    # Dropping columns with low corelation
    df = df.drop("depth", axis=1)
    df = df.drop("table", axis=1)
    y = np.asarray(df["price"])
    df = df.drop("price", axis=1)
    scaler = StandardScaler()
    encoder = OneHotEncoder()
    num_attr=["carat", "x", "y", "z"]
    cat_attr=["cut", "color", "clarity"]
    transformer = ColumnTransformer([
        ("num", scaler, num_attr),
        ("cat", encoder,cat_attr),
    ])
    x = transformer.fit_transform(df)
    train_linear(x, y)
    train_svm(x,y)
    train_elastic(x,y)
    train_sgd(x, y)
    train_xgb(x, y)
    return 0


def train_svm(x, y):

    svm = LinearSVR(C=0.2)
    scores = cross_validate(svm, x, y, scoring="neg_mean_squared_error", verbose=2, cv=5, return_train_score=True,
                            return_estimator=True)
    print("SVM train: " + str(scores['train_score']))
    print("SVM test: " + str(scores['test_score']))
    return 0


def train_linear(x, y):
    poly = PolynomialFeatures(2)
    x_poly = poly.fit_transform(x)
    linear_reg = LinearRegression()
    scores = cross_validate(linear_reg, x_poly, y, scoring="neg_mean_squared_error", verbose=2, cv=5, return_train_score=True,
                            return_estimator=True)
    print("Linear train: " + str(scores['train_score']))
    print("Linear test: " + str(scores['test_score']))
    return 0


def train_elastic(x, y):
    elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
    scores = cross_validate(elastic_net, x, y, scoring="neg_mean_squared_error", verbose=2, cv=5, return_train_score=True,
                            return_estimator=True)
    print("Elastic train: " + str(scores['train_score']))
    print("Elastic test: " + str(scores['test_score']))
    return 0


def train_sgd(x, y):
    sgd_reg = SGDRegressor()
    scores = cross_validate(sgd_reg, x, y, scoring="neg_mean_squared_error", verbose=2, cv=5, return_train_score=True,
                            return_estimator=True)
    print("SGD train: " + str(scores['train_score']))
    print("SGD test: " + str(scores['test_score']))
    return 0


def train_xgb(x,y):
    reg = xgboost.XGBRegressor(learning_rate=0.15, n_estimators=200)
    scores = cross_validate(reg, x, y, scoring="neg_mean_squared_error", verbose=2, cv=5, return_train_score=True,
                            return_estimator=True)
    print("XGB train: " + str(scores['train_score']))
    print("XGB test: " + str(scores['test_score']))


if __name__ == '__main__':
    # eda("train.csv")
    preprocess("train.csv")



