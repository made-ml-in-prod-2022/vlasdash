import unittest
from sklearn.ensemble import RandomForestClassifier

from models.train_model import (
    train_model,
    evaluate_model,
    serialize_model,
)
from models.predict_model import deserialize_model
from entities import (
    TrainingParams,
    MetricParams,
    ModelForestParams,
)
from tests.generate_dataset import generate_dataset


class TestTrainModel(unittest.TestCase):
    def setUp(self) -> None:
        self.categorical = [
            "cp",
            "restecg",
            "slope",
            "ca",
            "thal",
            "sex",
            "fbs",
            "exang"
        ]
        self.numerical = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        self.target = "condition"
        self.train_params = TrainingParams(
            model="RandomForestClassifier",
            model_forest_params=ModelForestParams(),
        )
        self.dataset = generate_dataset(100)
        self.metric_params = MetricParams(
            precision=True,
            recall=True,
            f1=False,
        )

    def test_train_model(self) -> None:
        target = self.dataset[self.target]
        del self.dataset[self.target]
        model = train_model(self.dataset, target, self.train_params)

        self.assertIsInstance(model, RandomForestClassifier)
        self.assertEqual(model.predict(self.dataset).shape, target.shape)

    def test_evaluate_model(self):
        target = self.dataset[self.target]
        del self.dataset[self.target]
        model = train_model(self.dataset, target, self.train_params)
        predicts = model.predict(self.dataset)
        metrics = evaluate_model(predicts, target, self.metric_params)

        self.assertEqual(len(metrics), 2)
        self.assertTrue("precision" in metrics)
        self.assertTrue("recall" in metrics)
        self.assertTrue("f1" not in metrics)
        self.assertTrue(0 <= metrics["precision"] <= 1)
        self.assertTrue(0 <= metrics["recall"] <= 1)

    def test_serialize_model(self):
        model = RandomForestClassifier(n_estimators=20)
        expected_output = "tests/resource/test_model.plk"
        real_output = serialize_model(model, expected_output)

        self.assertEqual(real_output, expected_output)
        model = deserialize_model(expected_output)
        self.assertIsInstance(model, RandomForestClassifier)


if __name__ == "__main__":
    unittest.main()
