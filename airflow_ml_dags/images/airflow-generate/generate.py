import os
import click
import numpy as np
import pandas as pd


def generate_data(data_size: int) -> pd.DataFrame:
    """Generate dataset for test
    :param data_size: size of generate dataset
    :return: generated dataset
    """

    np.random.seed(42)
    dataset = pd.DataFrame()

    dataset["cp"] = np.random.randint(0, 4, size=data_size)
    dataset["restecg"] = np.random.randint(0, 3, size=data_size)
    dataset["slope"] = np.random.randint(0, 3, size=data_size)
    dataset["ca"] = np.random.randint(0, 4, size=data_size)
    dataset["thal"] = np.random.randint(0, 3, size=data_size)
    dataset["sex"] = np.random.randint(0, 2, size=data_size)
    dataset["fbs"] = np.random.randint(0, 2, size=data_size)
    dataset["exang"] = np.random.randint(0, 2, size=data_size)
    dataset["age"] = np.random.randint(20, 90, size=data_size)
    dataset["trestbps"] = np.random.randint(60, 200, size=data_size)
    dataset["chol"] = np.random.randint(120, 600, size=data_size)
    dataset["thalach"] = np.random.randint(70, 200, size=data_size)
    dataset["oldpeak"] = np.round(
        np.random.uniform(0, 7, size=data_size),
        1
    )
    dataset["condition"] = np.random.randint(0, 2, size=data_size)

    return dataset


@click.command("create")
@click.option("--output-dir")
@click.option("--data-size", default=100)
def create_data(output_dir: str, data_size: int):
    dataset = generate_data(data_size)

    os.makedirs(output_dir, exist_ok=True)
    dataset.drop("condition", axis=1).to_csv(
        os.path.join(output_dir, "data.csv"),
        index=False,
    )
    dataset["condition"].to_csv(
        os.path.join(output_dir, "target.csv"),
        index=False,
    )


if __name__ == '__main__':
    create_data()
