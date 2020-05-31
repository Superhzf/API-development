# Standard build-in libraries
from dataclasses import dataclass
from typing import Optional
import os

# Related third party libraries
import pandas as pd
from pandas import DataFrame

# Local application/library specific imports


@dataclass
class Tables:
    table: DataFrame


TABLES: Optional[Tables] = None


def init_db() -> None:
    global TABLES
    test_df = pd.read_csv("./VirtualDataWarehouse/application_test.csv")
    TABLES = Tables(table=test_df)
    return None
