# Standard build-in libraries
from dataclasses import dataclass
import time
from typing import Union

# Related third party libraries
from fastapi import APIRouter
from fastapi import Depends
import numpy as np
from starlette.responses import JSONResponse
from sqlalchemy.orm import Session

# Local application/library specific imports
from ..models.api.lookup.input import DefaultPredictionRequestInput
from ..models.api.lookup.output import ApiDefaultPredictionRequestOutputPrediction

# from ..models.db import DefaultPredictionRequest
from ..models.api.lookup.output import ApiOutputNotFound
from ..utils.db import get_db
from ..utils.logging import logger
from ..utils.logging import LoggingType
from ..models.db import ModelMetaData
from ..prediction import model_settings
from ..utils.db import save_income_request_prediction


router = APIRouter()


@dataclass
class DefaultRequestRepr:
    api: DefaultPredictionRequestInput


def get_default_prediction(
        default_request_repr: DefaultRequestRepr, db: Session
) -> ApiDefaultPredictionRequestOutputPrediction:
    response = model_settings.CREDIT_MODEL.model.predict(request_input=default_request_repr.api)
    if isinstance(response, ApiOutputNotFound):
        response = -1
    response = ApiDefaultPredictionRequestOutputPrediction(predicted_default_probability=response,
                                                           loan_id=default_request_repr.api.SK_ID_CURR)
    model_metadata: ModelMetaData = model_settings.CREDIT_MODEL.metadata
    save_income_request_prediction(response, model_metadata, db)

    return response


@router.post("/default-prediction",
             response_model=ApiDefaultPredictionRequestOutputPrediction,
             summary="Predict people's repayment ability")
def request_default(request: DefaultPredictionRequestInput, db: Session = Depends(get_db)) -> JSONResponse:
    start_time = time.time()
    db.expire_all()
    default_request_repr = DefaultRequestRepr(api=request)

    response = get_default_prediction(default_request_repr, db)
    process_time = time.time() - start_time
    logger.info(
        "default-request",
        data={"request": request.dict(), "response": response.dict(), "process_time": process_time},
        type=LoggingType.REQUEST_RESPONSE_LOG,
    )
    return JSONResponse(response.dict())
