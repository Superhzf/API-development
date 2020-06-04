# Standard build-in libraries
import time
from typing import Any, Callable

# Related third party libraries
from fastapi import FastAPI, Request

# Local application/library specific imports
from HomeCreditDefaultPrediction.prediction import data_settings
from .utils.logging import logging_setup, logger, LoggingType
from .utils.app import get_app_name, get_app_version
from .prediction import model_settings
from .routers import ops
from .routers import default
from .utils.db import SessionLocal

logging_setup()

app = FastAPI(title=get_app_name(),
              version=get_app_version(),
              description='This service is responsible for estimating applicants ability to repay a loan')


@app.on_event('startup')
def startup_event() -> None:
    model_settings.init_models()
    data_settings.init_db()
    return None


app.include_router(ops.router, tags=['ops'])
app.include_router(default.router, prefix="/income", tags=["income"])


@app.middleware('http')
async def add_process_time(request: Request, call_next: Callable) -> Any:
    start_time = time.time()
    response = await call_next(request)
    end_time = time.time()
    process_time = end_time - start_time
    response.headers['X-Process-Time'] = str(process_time)

    logger.info(
        'request-process-time',
        data={
            'request_url_path': request.url.path,
            'request_query_params': request.query_params,
            'request_path_params': request.path_params,
            'process_time': process_time,
        },
        type=LoggingType.REQUEST_RESPONSE_LOG
    )

    return response


@app.middleware("http")
async def db_session_middleware(request: Request, call_next: Callable) -> Any:
    try:
        request.state.db = SessionLocal()
        return await call_next(request)
    finally:
        request.state.db.close()
    return Response("Internal server error", status_code=500)
