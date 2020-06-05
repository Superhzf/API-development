# Standard build-in libraries
import os
from typing import Any

# Related third party libraries
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DatabaseError
from starlette.requests import Request
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker


# Local application/library specific imports
from ..models.api.lookup.output import ApiDefaultPredictionRequestOutputPrediction
from ..models.db import DefaultPrediction
# from ..models.db import DefaultPredictionRequest
from ..models.db import ModelMetaData
from ..utils.logging import logger
from ..utils.logging import LoggingType


DB_USER = os.environ['POSTGRES_USER']
DB_PASSWORD = os.environ['POSTGRES_PASSWORD']
DB_HOST = os.environ['POSTGRES_HOST']
DB_NAME = os.environ['POSTGRES_DB']

SQLALCHEMY_DATABASE_URI: str = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
SANITIZED_SQLALCHEMY_DATABASE_URI: str = f"postgresql+psycopg2://{DB_HOST}/{DB_NAME}"
engine: Engine = create_engine(
    SQLALCHEMY_DATABASE_URI, pool_size=5, max_overflow=10, connect_args={'connect_timeout': 10}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=True, bind=engine)


# dependency in order to get access to db
def get_db(request: Request) -> Any:
    return request.state.db


def save_model_metadata(meta: ModelMetaData) -> ModelMetaData:
    """Store or retrieve the metadata for the prediction model"""
    db = SessionLocal()
    mm: ModelMetaData = db.query(ModelMetaData).filter_by(model_version=meta.model_version).first()
    try:
        if mm:
            logger.info(
                "model meta already exists",
                data={
                    "id": mm.id,
                    "created": mm.created_at,
                    "description": mm.model_description,
                    "version": mm.model_version
                },
                type=LoggingType.DB_OPS
            )
            return mm
        else:
            db.add(meta)
            db.commit()
            db.refresh(meta)
            logger.info(
                "model meta was saved",
                data={
                    "id": meta.id,
                    "created": meta.created_at,
                    "description": meta.model_description,
                    "version": meta.model_version,
                },
                type=LoggingType.DB_OPS
            )
            return meta
    except Exception as e:
        logger.info(e,
                    data={"id": meta.id,
                          "created": meta.created_at,
                          "version": meta.model_version,
                          "description": meta.model_description},
                    type=LoggingType.EXCEPTION)
    finally:
        db.close()


def prediction_api_2_db(
        response: ApiDefaultPredictionRequestOutputPrediction,
        model_metadata: ModelMetaData
) -> DefaultPrediction:
    """Convert the Prediction data model from API to DB"""
    response = DefaultPrediction.from_api(response=response,
                                          model_metadata=model_metadata)

    return response


def save_default_request_prediction(
        response: ApiDefaultPredictionRequestOutputPrediction,
        model_metadata: ModelMetaData,
        db: Session
) -> int:
    """Store the income request inputs and outputs in the db from from the prediction logic"""
    response = prediction_api_2_db(response, model_metadata)
    db.add(response)
    db.commit()
    db.refresh(response)
    return response
