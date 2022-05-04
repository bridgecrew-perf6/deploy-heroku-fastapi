"""
Common functions module test
"""
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
import pytest
import pickle
import train_model


@pytest.fixture
def data():
    """
    Get dataset
    """
    df = pd.read_csv("data/census_clean.csv")
    return df


def test_process_data(data):
    """
    Check split have same number of rows for X and y
    """
    encoder = pickle.loads("models/encoder.pkl")
    lb = pickle.loads("models/lb.pkl")

    X_test, y_test, _, _ = train_model.process_data(
        data,
        categorical_features= train_model.get_cat_features(),
        label="salary", encoder=encoder, lb=lb, training=False)

    assert len(X_test) == len(y_test)


def test_process_encoder(data):
    """
    Check split have same number of rows for X and y
    """
    encoder_test = pickle.loads("models/encoder.pkl")
    lb_test = pickle.load("models/lb.pkl")

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    _, _, encoder, lb = train_model.process_data(
        data,
        categorical_features=cat_features,
        label="salary", training=True)

    _, _, _, _ = train_model.process_data(
        data,
        categorical_features=cat_features,
        label="salary", encoder=encoder_test, lb=lb_test, training=False)

    assert encoder.get_params() == encoder_test.get_params()
    assert lb.get_params() == lb_test.get_params()


def test_inference_above():
    """
    Check inference performance
    """
    model = pickle.loads("models/model.pkl")
    encoder = pickle.loads("models/encoder.pkl")
    lb = pickle.loads("models/lb.pkl")

    array = np.array([[
                     32,
                     "Private",
                     "Some-college",
                     "Married-civ-spouse",
                     "Exec-managerial",
                     "Husband",
                     "Black",
                     "Male",
                     80,
                     "United-States"
                     ]])
    df_temp = DataFrame(data=array, columns=[
        "age",
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "hours-per-week",
        "native-country",
    ])

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X, _, _, _ = train_model.process_data(
                df_temp,
                categorical_features=cat_features,
                encoder=encoder, lb=lb, training=False)
    pred = train_model.inference(model, X)
    y = lb.inverse_transform(pred)[0]
    assert y == ">50K"


def test_inference_below():
    """
    Check inference performance
    """
    model = pickle.loads("models/model.pkl")
    encoder = pickle.loads("models/encoder.pkl")
    lb = pickle.loads("models/lb.pkl")

    array = np.array([[
                     19,
                     "Private",
                     "HS-grad",
                     "Never-married",
                     "Own-child",
                     "Husband",
                     "Black",
                     "Male",
                     40,
                     "United-States"
                     ]])
    df_temp = DataFrame(data=array, columns=[
        "age",
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "hours-per-week",
        "native-country",
    ])

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X, _, _, _ = train_model.process_data(
                df_temp,
                categorical_features=cat_features,
                encoder=encoder, lb=lb, training=False)
    pred = train_model.inference(model, X)
    y = lb.inverse_transform(pred)[0]
    assert y == "<=50K"