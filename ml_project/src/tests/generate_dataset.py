import numpy as np
import pandas as pd


def generate_dataset(dataset_size: int) -> pd.DataFrame:
    np.random.seed(42)
    dataset = pd.DataFrame()

    dataset["cp"] = np.random.randint(0, 4, size=dataset_size)
    dataset["restecg"] = np.random.randint(0, 3, size=dataset_size)
    dataset["slope"] = np.random.randint(0, 3, size=dataset_size)
    dataset["ca"] = np.random.randint(0, 4, size=dataset_size)
    dataset["thal"] = np.random.randint(0, 3, size=dataset_size)
    dataset["sex"] = np.random.randint(0, 2, size=dataset_size)
    dataset["fbs"] = np.random.randint(0, 2, size=dataset_size)
    dataset["exang"] = np.random.randint(0, 2, size=dataset_size)
    dataset["age"] = np.random.randint(20, 90, size=dataset_size)
    dataset["trestbps"] = np.random.randint(60, 200, size=dataset_size)
    dataset["chol"] = np.random.randint(120, 600, size=dataset_size)
    dataset["thalach"] = np.random.randint(70, 200, size=dataset_size)
    dataset["oldpeak"] = np.round(np.random.uniform(0, 7, size=dataset_size), 1)
    dataset["condition"] = np.random.randint(0, 2, size=dataset_size)

    return dataset
