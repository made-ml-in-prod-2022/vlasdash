import unittest
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from features.build_features import (
    get_target,
    build_categorical_pipeline,
    build_numerical_pipeline,
    build_transformer,
    make_features,
    serialize_transformer,
    deserialize_transformer,
)
from entities import FeatureParams
from tests.generate_dataset import generate_dataset


class TestFeatures(unittest.TestCase):

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
        self.feature_params = FeatureParams(
            categorical=self.categorical,
            numerical=self.numerical,
            target=self.target,
        )
        self.dataset = generate_dataset(100)

    def test_get_target(self) -> None:
        target = get_target(self.dataset, self.feature_params)

        self.assertEqual(len(target), 100)
        self.assertIsInstance(target, pd.Series)

    def test_build_categorical_pipeline(self) -> None:
        categorical_pipeline = build_categorical_pipeline()

        self.assertEqual(len(categorical_pipeline.steps), 2)
        self.assertIsInstance(categorical_pipeline, Pipeline)

    def test_build_numerical_pipeline(self) -> None:
        num_pipeline = build_numerical_pipeline()

        self.assertEqual(len(num_pipeline.steps), 2)
        self.assertIsInstance(num_pipeline, Pipeline)

    def test_build_transformer(self) -> None:
        transformer = build_transformer(self.feature_params)

        self.assertIsInstance(transformer, ColumnTransformer)

    def test_make_features(self) -> None:
        transformer = build_transformer(self.feature_params)
        transformer.fit(self.dataset)
        features = make_features(transformer, self.dataset)

        self.assertEqual(features.shape[0], 100)
        self.assertEqual(features.shape[1], 28)
        self.assertIsInstance(features, pd.DataFrame)

    def test_serialize_transformer(self):
        transformer = build_transformer(self.feature_params)
        expected_output = "tests/resource/test_transformer.plk"
        real_output = serialize_transformer(transformer, expected_output)
        self.assertEqual(expected_output, real_output)

        transformer = deserialize_transformer(real_output)
        self.assertIsInstance(transformer, ColumnTransformer)


if __name__ == "__main__":
    unittest.main()
