from __future__ import annotations

# Standard build-in libraries


# Related third party libraries
from pydantic import BaseModel
from fastapi import Body
from pydantic import PositiveInt


# Local application/library specific imports


class DefaultPredictionRequestInput(BaseModel):
    SK_ID_CURR: PositiveInt = Body(..., description="Loan ID")

