# Standard build-in libraries
from typing import Any
import json

# Related third party libraries
import numpy as np

# Local application/library specific imports
from ..models.api.lookup.input import DefaultPredictionRequestInput


class PredictionModel(object):
    def __init__(self, model_path: str):
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str) -> Any:
        with open(model_path, 'r') as file:
            return json.load(file)

    def predict(self, request_input: DefaultPredictionRequestInput) -> float:
        prediction = request_input.AMT_INCOME_TOTAL*self.model['model_parameters']['AMT_INCOME_TOTAL']+\
                     request_input.CNT_CHILDREN*self.model['model_parameters']['CNT_CHILDREN']
        prediction = 1.0/(1+np.exp(-1*prediction))
        return prediction
