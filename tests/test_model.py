# Standard build-in libraries
import joblib

# Related third party libraries
import numpy as np

# Local application/library specific imports
from HomeCreditDefaultPrediction.models.api.lookup.output import ApiOutputNotFound
from HomeCreditDefaultPrediction.prediction.default_prediction import feature_engineering
from HomeCreditDefaultPrediction.prediction.default_prediction import get_default_request_data
from HomeCreditDefaultPrediction.prediction import data_settings
from HomeCreditDefaultPrediction.prediction.model_settings import MODEL_PATH


def test_model_is_available():
    model = joblib.load(MODEL_PATH)

    assert model["meta_data"]['version'] is not None, "model version is missing"
    assert model["meta_data"]['created_at'] is not None, "the timestamp when the model was created is missing"
    assert model["meta_data"]['model_description'] is not None, "the model description is missing"
    assert model["model"] is not None, "the model is missing"
    assert model["features"] is not None, "the feature file is missing"
    assert model["transformer"] is not None, "the transformer for feature engineering is missing"
    assert model["test_performance"] is not None, "the performance on the test set is missing"


def test_data_could_be_loaded():
    data_settings.init_db()
    table = data_settings.TABLES.table
    assert table.shape[1] > 0, "No data is found!"


def test_succeed_match_request_data_from_dw():
    data_settings.init_db()
    table = data_settings.TABLES.table
    pool_id = table['SK_ID_CURR'].values
    random_loan_id = np.random.choice(pool_id)
    record = get_default_request_data(random_loan_id)
    assert record.shape[0] == 1, "only 1 record should be found!"


def test_fail_match_request_data_from_dw():
    data_settings.init_db()
    table = data_settings.TABLES.table
    pool_id = table['SK_ID_CURR'].values
    random_loan_id = np.random.randint(low=1, high=pool_id.max())
    while random_loan_id in pool_id:
        random_loan_id = np.random.randint(low=1, high=pool_id.max())
    record = get_default_request_data(random_loan_id)
    assert isinstance(record, ApiOutputNotFound), "No record should be found"


def test_feature_engineering():
    data_settings.init_db()
    table = data_settings.TABLES.table
    pool_id = table['SK_ID_CURR'].values
    random_loan_id = np.random.choice(pool_id)
    record = get_default_request_data(random_loan_id)
    model = joblib.load(MODEL_PATH)
    full_record = feature_engineering(record, model['transformer'])
    assert full_record.shape[0]==1, "Only 1 record should be matched"

