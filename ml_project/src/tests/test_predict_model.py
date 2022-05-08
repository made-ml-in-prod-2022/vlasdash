import unittest
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from models.predict_model import predict_model, write_prediction
from tests.generate_dataset import generate_dataset


class TestPredictModel(unittest.TestCase):
    def setUp(self) -> None:
        self.target = "condition"
        self.dataset = generate_dataset(100)

    def test_predict_model(self):
        model = RandomForestClassifier(n_estimators=20)
        target = self.dataset[self.target]
        del self.dataset[self.target]
        model.fit(self.dataset, target)
        prediction = predict_model(model, self.dataset)

        self.assertEqual(prediction.shape, target.shape)
        self.assertIsInstance(prediction, np.ndarray)

    def test_serialize_prediction(self):
        model = RandomForestClassifier(n_estimators=20)
        target = self.dataset[self.target]
        del self.dataset[self.target]
        model.fit(self.dataset, target)
        prediction = predict_model(model, self.dataset)
        expected_output = "tests/resource/test_prediction.csv"
        real_output = write_prediction(prediction, expected_output)

        self.assertEqual(real_output, expected_output)


if __name__ == "__main__":
    unittest.main()
