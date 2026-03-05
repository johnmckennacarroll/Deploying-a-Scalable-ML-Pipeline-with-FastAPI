import pytest
import numpy as np

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

# TODO: implement the first test. Change the function name and input as needed
def test_one():
    """
    # Test that process_data returns X and y with matching row counts.
    """
    import pandas as pd

    data = pd.DataFrame({
        "age": [37, 50],
        "workclass": ["Private", "Self-emp-not-inc"],
        "fnlgt": [178356, 83311],
        "education": ["HS-grad", "Bachelors"],
        "education-num": [10, 13],
        "marital-status": ["Married-civ-spouse", "Married-civ-spouse"],
        "occupation": ["Prof-specialty", "Exec-managerial"],
        "relationship": ["Husband", "Husband"],
        "race": ["White", "White"],
        "sex": ["Male", "Male"],
        "capital-gain": [0, 0],
        "capital-loss": [0, 0],
        "hours-per-week": [40, 60],
        "native-country": ["United-States", "United-States"],
        "salary": ["<=50K", ">50K"]
    })

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

    X, y, encoder, lb = process_data(
        data,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )

    assert X.shape[0] == data.shape[0]
    assert y.shape[0] == data.shape[0]


# TODO: implement the second test. Change the function name and input as needed
def test_two():
    """
    # Test that train_model returns a trained RandomForestClassifier.
    """
    import pandas as pd

    data = pd.DataFrame({
        "age": [37, 50],
        "workclass": ["Private", "Self-emp-not-inc"],
        "education": [178356, 83311],
        "education": ["HS-grad", "Bachelors"],
        "education-num": [10, 13],
        "marital-status": ["Married-civ-spouse", "Married-civ-spouse"],
        "occupation": ["Prof-specialty", "Exec-managerial"],
        "relationship": ["Husband", "Husband"],
        "race": ["White", "White"],
        "sex": ["Male", "Male"],
        "capital-gain": [0, 0],
        "capital-loss": [40, 60],
        "native-country": ["United-States", "United-States"],
        "salary": ["<=50K", ">50K"]
    })

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

    X, y, _, _ = process_data(
        data,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )

    model = train_model(X, y)

    assert model is not None
    assert model.__class__.__name__ == "RandomForestClassifier"


# TODO: implement the third test. Change the function name and input as needed
def test_three():
    """
    # Test that compute_model_metrics returns values between 0 and 1.
    """
    y = np.array([0, 1, 1, 0])
    preds = np.array([0, 1, 1, 0])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1
