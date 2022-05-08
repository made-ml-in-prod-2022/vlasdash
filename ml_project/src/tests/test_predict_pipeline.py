import unittest

from tests.generate_dataset import generate_dataset
from predict_pipeline import predict_pipeline
from entities import (
    PredictPipelineParams,
    FeatureParams,
)
from features.build_features import (
    build_transformer,
    serialize_transformer,
)


class TestPredictPipeline(unittest.TestCase):

    def test_predict_pipeline(self):
        dataset = generate_dataset(100)
        path = "tests/resource/test_dataset.csv"
        dataset.to_csv(path, index=False)
        model_path = "tests/resource/test_model.plk"
        transformer_path = "tests/resource/test_transformer.plk"
        expected_prediction_path = "tests/resource/test_prediction.csv"
        params = PredictPipelineParams(
            input_data_path=path,
            model_path=model_path,
            transformer_path=transformer_path,
            prediction_path=expected_prediction_path,
        )
        categorical = ["cp", "restecg", "slope", "ca", "thal", "sex",
                            "fbs", "exang"]
        numerical = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        target = "condition"
        feature_params = FeatureParams(
            categorical=categorical,
            numerical=numerical,
            target=target,
        )
        transformer = build_transformer(feature_params)
        transformer.fit(dataset)
        serialize_transformer(
            transformer,
            transformer_path,
        )

        real_prediction_path = predict_pipeline(params)

        self.assertEqual(expected_prediction_path, real_prediction_path)


if __name__ == "__main__":
    unittest.main()
