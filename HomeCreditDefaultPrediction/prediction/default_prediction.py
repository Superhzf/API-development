# Standard build-in libraries
import gc
import joblib
import os
from typing import Union

# Related third party libraries
import pandas as pd
from pandas import DataFrame

# Local application/library specific imports
from ..models.api.lookup.input import DefaultPredictionRequestInput
from HomeCreditDefaultPrediction.prediction.data_settings import TABLES
from ..models.api.lookup.output import ApiOutputNotFound
from ..models.api.lookup.output import NotFoundMessage
from .train_pipelines.utils import general_preprocess
from .train_pipelines.utils import _drop_application_columns

PREDICTION_FOLDER = './model_binaries/'


def get_default_request_data(loan_id: int) -> Union[DataFrame, ApiOutputNotFound]:
    table = TABLES.table
    record = table[table['SK_ID_CURR'] == loan_id]
    if len(record) == 0:
        return ApiOutputNotFound(message=NotFoundMessage.could_not_find_loan_id)
    else:
        return record


def feature_engineering(basic_feature: DataFrame) -> DataFrame:
    df = general_preprocess(basic_feature)
    transformer = joblib.load(os.path.join(PREDICTION_FOLDER, "transformer.joblib"))
    transformed_data = transformer.transform(df)
    transformed_data = _drop_application_columns(transformed_data)
    del transformer
    del df

    bureau_df = joblib.load(os.path.join(PREDICTION_FOLDER, "bureau_and_balance.joblib"))
    transformed_data = pd.merge(transformed_data, bureau_df, on='SK_ID_CURR', how='left')
    del bureau_df
    gc.collect()

    prev_df = joblib.load(os.path.join(PREDICTION_FOLDER, "previous.joblib"))
    transformed_data = pd.merge(transformed_data, prev_df, on='SK_ID_CURR', how='left')
    del prev_df
    gc.collect()

    pos_df = joblib.load(os.path.join(PREDICTION_FOLDER, "pos_cash.joblib"))
    transformed_data = pd.merge(transformed_data, pos_df, on='SK_ID_CURR', how='left')
    del pos_df
    gc.collect()

    ins_df = joblib.load(os.path.join(PREDICTION_FOLDER, "payments.joblib"))
    transformed_data = pd.merge(transformed_data, ins_df, on='SK_ID_CURR', how='left')
    del ins_df
    gc.collect()

    cc_df = joblib.load(os.path.join(PREDICTION_FOLDER, "credit_card.joblib"))
    transformed_data = pd.merge(transformed_data, cc_df, on='SK_ID_CURR', how='left')
    del cc_df
    gc.collect()

    transformed_data.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in transformed_data.columns]

    predictors = []
    with open(os.path.join(PREDICTION_FOLDER, "features.txt")) as f:
        for line in f:
            predictors.append(line.strip())

    transformed_data = transformed_data[predictors]
    return transformed_data[predictors]


class PredictionModel(object):
    def __init__(self, model_path: str):
        # self.model = self._load_model(model_path)
        self.model = joblib.load(model_path)

    # def _load_model(self, model_path: str) -> Any:
    #     with open(model_path, 'r') as file:
    #         return json.load(file)

    @staticmethod
    def predict(self, request_input: DefaultPredictionRequestInput) -> Union[float, ApiOutputNotFound]:
        basic_feature = get_default_request_data(request_input.SK_ID_CURR)
        if isinstance(basic_feature, ApiOutputNotFound):
            return basic_feature

        full_features = feature_engineering(basic_feature)
        prediction = self.model.predict_proba(full_features)[:, 1]
        # prediction = request_input.AMT_INCOME_TOTAL*self.model['model_parameters']['AMT_INCOME_TOTAL']+\
        #              request_input.CNT_CHILDREN*self.model['model_parameters']['CNT_CHILDREN']
        # prediction = 1.0/(1+np.exp(-1*prediction))

        return prediction
