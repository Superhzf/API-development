# Standard build-in libraries
from dataclasses import dataclass
from typing import Optional

# Related third party libraries

# Local application/library specific imports
from ..utils.logging import logger
from ..utils.logging import LoggingType
from ..models.db import ModelMetaData
from .default_prediction import PredictionModel
from ..utils.db import save_model_metadata


MODEL_PATH: str = "./HomeCreditDefaultPrediction/prediction/model_binaries/lgb_model.joblib"


@dataclass
class CreditModel:
    model: PredictionModel
    metadata: ModelMetaData


CREDIT_MODEL: Optional[CreditModel] = None


def init_models() -> None:
    global CREDIT_MODEL
    prediction_model = PredictionModel(MODEL_PATH)
    meta = prediction_model.model['metadata']
    model_metadata = save_model_metadata(
        ModelMetaData(model_version=meta['version'], model_description=meta['model_description'])
    )
    CREDIT_MODEL = CreditModel(model=PredictionModel(MODEL_PATH), metadata=model_metadata)
    logger.info("loading model", data={"model": meta}, type=LoggingType.STARTUP_MESSAGE)
