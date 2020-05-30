from __future__ import annotations
# Standard build-in libraries
from typing import Any

# Related third party libraries
from sqlalchemy.orm import backref
from sqlalchemy import BigInteger
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy.sql import func
from sqlalchemy import Index
from sqlalchemy import Integer
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy import String

# Local application/library specific imports
from .api.lookup.input import DefaultPredictionRequestInput
from .api.lookup.output import ApiDefaultPredictionRequestOutputPrediction

Base: Any = declarative_base()


# class DefaultPredictionRequest(Base):
#     __tablename__ = "default_requests"
#     __table_args__ = {"comment": "The logged requests of the home credit default prediction endpoint."}
#     id = Column(BigInteger, primary_key=True, comment="The primary key for the table")
#     created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="When the record was inserted")
#     raw_request = Column(JSONB, nullable=False, comment="The raw request. It can be used to replay requests for debugging purpose")
#     SK_ID_CURR = Column(BigInteger, nullable=False, comment="Loan ID")
#     CNT_CHILDREN = Column(Integer, comment="The number of children the client has")
#     AMT_INCOME_TOTAL = Column(Integer,comment="Income of the client")
#
#     @classmethod
#     def from_api(cls, request: DefaultPredictionRequestInput) -> DefaultPredictionRequest:
#         return DefaultPredictionRequest(
#             raw_request=request.dict(),
#             SK_ID_CURR=request.SK_ID_CURR,
#             CNT_CHILDREN=request.CNT_CHILDREN,
#             AMT_INCOME_TOTAL=request.AMT_INCOME_TOTAL,
#         )


class DefaultPrediction(Base):
    __tablename__ = "default_predictions"
    id = Column(BigInteger, primary_key=True, comment="The primary key for the table.")
    create_at = Column(DateTime(timezone=True), server_default=func.now(), comment="When the record was inserted in the table.")
    request_id = Column(
        BigInteger,
        ForeignKey("default_requests.id"),
        nullable=False,
        comment="Foreign key that references default_requests.id."
    )
    model_id = Column(
        Integer,
        ForeignKey("model_metadata.id"),
        nullable=False,
        comment="Foreign key that references model_metadata.id."
    )
    default_prob = Column(Float, nullable=False, comment="The default probability by the model")
    # default_request = relationship(
    #     "DefaultPredictionRequest", backref=backref("default_requests", uselist=False, cascade="all, delete")
    # )

    @classmethod
    def from_api(
            cls,
            # default_req: DefaultPredictionRequest,
            response: ApiDefaultPredictionRequestOutputPrediction,
            model_metadata: ModelMetaData
    ) -> DefaultPrediction:
        return DefaultPrediction(
            default_prob=response.predicted_default_probability,
            # default_request=default_req,
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
