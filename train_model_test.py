"""
Common functions module test
"""
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
import pytest
import pickle
import train_model
import pandas as pd


@pytest.fixture
def data():
    """
    Get dataset
    """
    df = pd.read_csv("data/census_clean.csv")
    return df

def test_get_data():
    data = train_model.get_data()

    assert type(data) == pd.DataFrame
    assert data.empty == False

def test_process_data(data):
    """
    Check split have same number of rows for X and y
    """
    encoder = pickle.load(open("models/encode.pkl", 'rb'))
    lb = pickle.load(open("models/lb.pkl", 'rb'))

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

    X_test, y_test, _, _ = train_model.process_data(
        data,
        categorical_features= cat_features,
        label="salary", encoder=encoder, lb=lb, training=False)

    assert len(X_test) == len(y_test)


def test_process_encoder(data):
    """
    Check split have same number of rows for X and y
    """
    encoder_test = pickle.load(open("models/encode.pkl", 'rb'))
    lb_test = pickle.load(open("models/lb.pkl", 'rb'))

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