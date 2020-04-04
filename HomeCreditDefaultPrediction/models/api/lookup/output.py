from __future__ import annotations

# Standard build-in libraries


# Related third party libraries
from pydantic import BaseModel
from fastapi import Body

# Local application/library specific imports


class ApiDefaultPredictionRequestOutputPrediction(BaseModel):
    predicted_default_probability: float = Body(..., description="Prediction for the ability to pay", ge=0, le=1, example=0.5)

