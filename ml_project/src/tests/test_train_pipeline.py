import unittest
from tests.generate_dataset import generate_dataset

from train_pipeline import train_pipeline
from entities import (
    TrainingPipelineParams,
    TrainingParams,
    SplittingParams,
    MetricParams,
    FeatureParams,
    ModelSGDParams,
)


class TestTrainPipeline(unittest.TestCase):

    def test_train_pipeline(self):
        dataset = generate_dataset(100)
        path = "tests/resource/test_dataset.csv"
        dataset.to_csv(path, index=False)
        expected_model_path = "tests/resource/test_model.plk"
        expected_metric_path = "tests/resource/test_metrics.json"
        expected_transformer_path = "tests/resource/test_transformer.plk"
        categorical = ["cp", "restecg", "slope", "ca", "thal", "sex", "fbs", "exang"]
        numerical = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        target = "condition"
        params = TrainingPipelineParams(
            input_data_path=path,
            output_model_path=expected_model_path,
            metric_path=expected_metric_path,
            transformer_path=expected_transformer_path,
            splitting_params=SplittingParams(),
            feature_params=FeatureParams(
                numerical=numerical,
                categorical=categorical,
                target=target,
            ),
            train_params=TrainingParams(
                model="SGDClassifier",
                model_sgd_params=ModelSGDParams(),
            ),
            metric_params=MetricParams(
                precision=True,
                recall=True,
                f1=False
            ),
        )
        real_model_path, real_transformer_path, metrics = train_pipeline(params)

        self.assertEqual(expected_model_path, real_model_path)
        self.assertEqual(expected_transformer_path, real_transformer_path)
        self.assertEqual(len(metrics), 2)
        self.assertTrue(0 <= metrics["precision"] <= 1)
        self.assertTrue(0 <= metrics["recall"] <= 1)


if __name__ == "__main__":
    unittest.main()
