import unittest
import pandas as pd

from data.make_dataset import load_dataset, split_dataset
from entities import SplittingParams
from tests.generate_dataset import generate_dataset


class TestDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = generate_dataset(100)
        self.path = "tests/resource/test_dataset.csv"
        self.dataset.to_csv(self.path, index=False)

    def test_load_dataset(self) -> None:
        data = load_dataset(self.path)
        self.assertEqual(data.shape[0], 100)
        self.assertEqual(data.shape[1], 14)
        self.assertIsInstance(data, pd.DataFrame)

    def test_split_dataset(self) -> None:
        splitting_params = SplittingParams(
            valid_size=0.2,
            random_state=42,
            shuffle=True
        )
        train_df, valid_df = split_dataset(self.dataset, splitting_params)

        self.assertEqual(train_df.shape[0], 80)
        self.assertEqual(train_df.shape[1], 14)
        self.assertEqual(valid_df.shape[0], 20)
        self.assertEqual(valid_df.shape[1], 14)


if __name__ == "__main__":
    unittest.main()
