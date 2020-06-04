# Standard build-in libraries
from datetime import datetime
import gc
import joblib
import os
from typing import Union

# Related third party libraries
import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline

# Local application/library specific imports
from HomeCreditDefaultPrediction.prediction import data_settings
from HomeCreditDefaultPrediction.utils.logging import logger
from ..models.api.lookup.input import DefaultPredictionRequestInput
from ..models.api.lookup.output import ApiOutputNotFound
from ..models.api.lookup.output import NotFoundMessage
from .train_pipelines.utils import general_preprocess
from .train_pipelines.utils import _drop_application_columns

DATA_FOLDER = './VirtualDataWarehouse/'


def get_default_request_data(loan_id: int) -> Union[DataFrame, ApiOutputNotFound]:
    table = data_settings.TABLES.table
    record = table[table['SK_ID_CURR'] == loan_id]
    if len(record) == 0:
        return ApiOutputNotFound(message=NotFoundMessage.could_not_find_loan_id)
    else:
        return record


def feature_engineering(basic_feature: DataFrame, transformer: Pipeline) -> DataFrame:
    df = general_preprocess(basic_feature)
    transformed_data = transformer.transform(df)
    transformed_data = _drop_application_columns(transformed_data)
    del transformer
    del df

    bureau_df = joblib.load(os.path.join(DATA_FOLDER, "bureau_and_balance.joblib"))
    transformed_data = pd.merge(transformed_data, bureau_df, on='SK_ID_CURR', how='left')
    del bureau_df
    gc.collect()

    prev_df = joblib.load(os.path.join(DATA_FOLDER, "previous.joblib"))
    transformed_data = pd.merge(transformed_data, prev_df, on='SK_ID_CURR', how='left')
    del prev_df
    gc.collect()

    pos_df = joblib.load(os.path.join(DATA_FOLDER, "pos_cash.joblib"))
    transformed_data = pd.merge(transformed_data, pos_df, on='SK_ID_CURR', how='left')
    del pos_df
    gc.collect()

    ins_df = joblib.load(os.path.join(DATA_FOLDER, "payments.joblib"))
    transformed_data = pd.merge(transformed_data, ins_df, on='SK_ID_CURR', how='left')
    del ins_df
    gc.collect()

    cc_df = joblib.load(os.path.join(DATA_FOLDER, "credit_card.joblib"))
    transformed_data = pd.merge(transformed_data, cc_df, on='SK_ID_CURR', how='left')
    del cc_df
    gc.collect()

    transformed_data.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in transformed_data.columns]

    transformed_data = transformed_data
    return transformed_data


class PredictionModel(object):
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)

    def predict(self, request_input: DefaultPredictionRequestInput) -> Union[float, ApiOutputNotFound]:
        try:
            basic_feature = get_default_request_data(request_input.SK_ID_CURR)
        except Exception as e:
            logger.error('Fail matching the basic information from DW!', data={'loan id': request_input.SK_ID_CURR,
                                                                               'time': datetime.utcnow()})
        if isinstance(basic_feature, ApiOutputNotFound):
            return basic_feature
        try:
            full_features = feature_engineering(basic_feature, self.model['transformer'])[self.model["features"]]
        except Exception as e:
            logger.error("Fail matching full features from DW!", data={'loan id': request_input.SK_ID_CURR,
                                                                       'time': datetime.utcnow()})

        try:
            prediction = self.model['model'].predict_proba(full_features)[:, 1]
        except Exception as e:
            logger.error('Fail making predictions!', data={'loan id': request_input.SK_ID_CURR,
                                                           'time': datetime.utcnow()})
        return prediction
