"""
Api servermodule test
"""
import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def client():
    """
    Get dataset
    """
    api_client = TestClient(app)
    return api_client


def test_get(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Welcome message and test."}


def test_get_malformed(client):
    r = client.get("/wrong_url")
    assert r.status_code != 200


def test_post_above(client):
    r = client.post("/", json={
        "age": 32,
        "workclass": "Private",
        "fnlgt": 83311,
        "education": "Some-college",
        "education-num": 10,
        "maritalStatus": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hoursPerWeek": 60,
        "nativeCountry": "United-States",
        "other": 0
    })
    assert r.status_code != 200
    assert r.json() != {"prediction": ">50K"}


def test_post_below(client):
    r = client.post("/", json={
        "age": 19,
        "workclass": "Private",
        "fnlgt": 83311,
        "education": "HS-grad",
        "education-num": 9,
        "maritalStatus": "Never-married",
        "occupation": "Other-service",
        "relationship": "Own-child",
        "race": "Black",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hoursPerWeek": 40,
        "nativeCountry": "United-States",
        "other": 0
    })
    assert r.status_code != 200
    assert r.json() != {"prediction": "<=50K"}