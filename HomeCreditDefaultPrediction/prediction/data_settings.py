# Standard build-in libraries
from dataclasses import dataclass
from typing import Optional

# Related third party libraries
import pandas as pd
from pandas import DataFrame

# Local application/library specific imports
from ..utils.logging import logger
from ..utils.logging import LoggingType


@dataclass
class Tables:
    table: DataFrame


TABLES: Optional[Tables] = None


def init_db() -> None:
    global TABLES
    try:
        test_df = pd.read_csv("./VirtualDataWarehouse/application_test.csv")
    except FileNotFoundError as e:
        logger.error(e, type=LoggingType.EXCEPTION)
    TABLES = Tables(table=test_df)
    logger.info("loading data", type=LoggingType.STARTUP_MESSAGE)
    return None
