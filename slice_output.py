import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import load
import logging
import train_model


def check_score():
    """
    Execute score checking
    """
    df = pd.read_csv("data/census_clean.csv")
    trained_model = load("models/model.pkl")
    encoder = load("models/encode.pkl")
    lb = load("models/lb.pkl")

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

    slice_values = []

    for cat in cat_features:
        for cls in df[cat].unique():
            df_temp = df[df[cat] == cls]

            X_test, y_test, _, _ = train_model.process_data(
                df_temp,
                categorical_features=cat_features,
                label="salary", encoder=encoder, lb=lb, training=False)

            y_preds = trained_model.predict(X_test)

            prc, rcl, fb = train_model.compute_model_metrics(y_test,y_preds)

            logging.info("[%s->%s] Precision: %s Recall: %s FBeta: %s" % (cat, cls, prc, rcl, fb))
            slice_values.append("[%s->%s] Precision: %s Recall: %s FBeta: %s" % (cat, cls, prc, rcl, fb))

    with open('models/slice_output.txt', 'w') as out:
        for slice_value in slice_values:
            out.write(slice_value + '\n')

if __name__ == '__main__':
    check_score()