from __future__ import annotations
# Standard build-in libraries
from typing import Any

# Related third party libraries
from sqlalchemy import BigInteger
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy.sql import func
from sqlalchemy import Index
from sqlalchemy import Integer
from sqlalchemy import String

# Local application/library specific imports
from .api.lookup.output import ApiDefaultPredictionRequestOutputPrediction

Base: Any = declarative_base()


class DefaultPrediction(Base):
    __tablename__ = "default_predictions"
    id = Column(BigInteger, primary_key=True, comment="The primary key for the table.")
    SK_ID_CURR = Column(BigInteger, comment="Loan ID")
    create_at = Column(DateTime(timezone=True), server_default=func.now(), comment="When the record was inserted in the table.")
    model_id = Column(
        Integer,
        ForeignKey("model_metadata.id"),
        nullable=False,
        comment="Foreign key that references model_metadata.id."
    )
    default_prob = Column(Float, nullable=False, comment="The default probability by the model")

    @classmethod
    def from_api(
            cls,
            response: ApiDefaultPredictionRequestOutputPrediction,
            model_metadata: ModelMetaData
    ) -> DefaultPrediction:
        return DefaultPrediction(
            default_prob=response.predicted_default_probability,
            SK_ID_CURR=response.loan_id,
            model_id=model_metadata.id,
        )


class ModelMetaData(Base):
    __tablename__ = "model_metadata"
    __table_args__ = (
        Index("model_version_index", "model_version"),
        {"comment": "Information about models that have been deployed."},
    )
    id = Column(Integer, primary_key=True, comment="The primary key for the table.")
    created_at = Column(
        DateTime(timezone=True),server_default=func.now(), comment="When the record was inserted in the table."
    )
    model_version = Column(String(100), nullable=False, unique=True, comment="The version of the model")
    model_description = Column(String(100), nullable=False, comment="The description of the model.")
