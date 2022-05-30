import requests


def test_predict_correct():
    """Test correct work online inference predictions."""

    request = [
      {
        "id": 0,
        "age": 69,
        "sex": 1,
        "cp": 0,
        "trestbps": 160,
        "chol": 234,
        "fbs": 1,
        "restecg": 2,
        "thalach": 131,
        "exang": 0,
        "oldpeak": 0.1,
        "slope": 1,
        "ca": 1,
        "thal": 0,
      },
      {
        "id": 1,
        "age": 52,
        "sex": 1,
        "cp": 0,
        "trestbps": 152,
        "chol": 298,
        "fbs": 1,
        "restecg": 0,
        "thalach": 178,
        "exang": 0,
        "oldpeak": 1.2,
        "slope": 1,
        "ca": 0,
        "thal": 2,
      },
    ]

    response = requests.get(
        "http://0.0.0.0:8000/predict/",
        json=request,
    )

    assert response.status_code == 200
    assert len(response.json()) == len(request)
    assert response.json()[0]["id"] == request[0]["id"]
    assert response.json()[1]["id"] == request[1]["id"]


def test_predict_incorrect_data():
    """Test work predictions with incorrect thal value."""

    request = [
      {
        "id": 0,
        "age": 69,
        "sex": 1,
        "cp": 0,
        "trestbps": 160,
        "chol": 234,
        "fbs": 1,
        "restecg": 2,
        "thalach": 131,
        "exang": 0,
        "oldpeak": 0.1,
        "slope": 1,
        "ca": 1,
        "thal": 5,
      },
    ]

    response = requests.get(
        "http://0.0.0.0:8000/predict/",
        json=request,
    )

    assert response.status_code == 400
