# Standard build-in libraries
import os
from typing import Any

# Related third party libraries
from starlette.requests import Request
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

# Local application/library specific imports


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

