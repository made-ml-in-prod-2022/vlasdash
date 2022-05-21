from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from entities.splitting_params import SplittingParams


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def split_dataset(
    data: pd.DataFrame, params: SplittingParams
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data, valid_data = train_test_split(
        data, test_size=params.valid_size,
        random_state=params.random_state,
        shuffle=params.shuffle
    )

    return train_data, valid_data
