import unittest
import numpy as np

from features.custom_transformer import CustomMinMaxScaler


class TestCustomMinMaxScaler(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = np.array([[1, 5, 7], [3, 7, 0], [2, 2, 2]])
        self.scaler = CustomMinMaxScaler()

    def test_fit(self) -> None:
        fit_scaler = self.scaler.fit(self.dataset)
        self.assertEqual(self.scaler.min.all(), np.array([1, 2, 0]).all())
        self.assertEqual(self.scaler.max.all(), np.array([3, 7, 7]).all())
        self.assertIsInstance(fit_scaler, CustomMinMaxScaler)

    def test_transform(self) -> None:
        self.scaler.fit(self.dataset)
        result = self.scaler.transform(self.dataset)

        self.assertEqual(
            result.all(),
            np.array([[0, 3/5, 1], [1, 1, 0], [0.5, 0, 2/7]]).all()
        )
        self.assertIsInstance(result, np.ndarray)


if __name__ == "__main__":
    unittest.main()
