# Standard build-in libraries
import joblib

# Related third party libraries


# Local application/library specific imports
from HomeCreditDefaultPrediction.prediction.model_settings import MODEL_PATH


def test_that_model_is_available():
    model = joblib.load(MODEL_PATH)

    assert model["meta_data"]['version'] is not None, "model version is missing"
    assert model["meta_data"]['created_at'] is not None, "the timestamp when the model was created is missing"
    assert model["meta_data"]['model_description'] is not None, "the model description is missing"
    assert model["model"] is not None, "the model is missing"
    assert model["features"] is not None, "the feature file is missing"
    assert model["transformer"] is not None, "the transformer for feature engineering is missing"
    assert model["test_performance"] is not None, "the performance on the test set is missing"
