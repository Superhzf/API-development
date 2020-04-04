from __future__ import annotations

# Standard build-in libraries


# Related third party libraries
from pydantic import BaseModel
from fastapi import Body
from pydantic import PositiveInt


# Local application/library specific imports


class DefaultPredictionRequestInput(BaseModel):
    SK_ID_CURR: PositiveInt = Body(..., description="Loan ID")
    CNT_CHILDREN: int = Body(..., description="The number of children the client has", ge=0)
    AMT_INCOME_TOTAL: int = Body(..., description="Income of the client", ge=0)
