import requests


data = {
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
    }
r = requests.post('https://udacity-project-mw.herokuapp.com/', json=data)

assert r.status_code == 200

print("Response code: %s" % r.status_code)
print("Response body: %s" % r.json())