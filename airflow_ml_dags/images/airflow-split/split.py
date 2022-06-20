import os
import pandas as pd
import click
from sklearn.model_selection import train_test_split


@click.command("split_dataset")
@click.option("--input-dir")
@click.option("--output-dir")
@click.option("--valid-size", default=0.2)
@click.option("--random-state", default=42)
@click.option("--shuffle", default=True)
def split_data(
        input_dir: str,
        output_dir: str,
        valid_size: int,
        random_state: int,
        shuffle: bool
):
    features = pd.read_csv(os.path.join(input_dir, "data.csv"))
    target = pd.read_csv(os.path.join(input_dir, "target.csv"))

    train_features, valid_features, train_target, valid_target \
        = train_test_split(
            features, target, test_size=valid_size,
            random_state=random_state,
            shuffle=shuffle,
        )

    os.makedirs(output_dir, exist_ok=True)
    train_features.to_csv(
        os.path.join(output_dir, "train_data.csv"),
        index=False
    )
    train_target.to_csv(
        os.path.join(output_dir, "train_target.csv"),
        index=False
    )
    valid_features.to_csv(
        os.path.join(output_dir, "valid_data.csv"),
        index=False
    )
    valid_target.to_csv(
        os.path.join(output_dir, "valid_target.csv"),
        index=False
    )


if __name__ == '__main__':
    split_data()
