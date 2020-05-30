from __future__ import annotations

# Standard build-in libraries


# Related third party libraries
from pydantic import BaseModel
from fastapi import Body

# Local application/library specific imports
from ...string import EstEnum


class NotFoundMessage(EstEnum):
    could_not_find_loan_id = "could-not-find-loan-id"


class ApiDefaultPredictionRequestOutputPrediction(BaseModel):
    predicted_default_probability: float = Body(..., description="Prediction for the ability to pay",
                                                ge=0, le=1, example=0.5)


class ApiOutputNotFound(BaseModel):
    message: NotFoundMessage = Body(description="Explain why the default prediction is not provided", default=None)
