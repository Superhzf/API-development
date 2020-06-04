# Standard build-in libraries
import datetime
from typing import Dict
import os
# Related third party libraries
from fastapi import APIRouter, Depends
import psutil
from sqlalchemy.orm import Session

# Local application/library specific imports
from ..utils.logging import logger, LoggingType
from ..utils.app import get_app_name, get_app_version
from HomeCreditDefaultPrediction.models.api.ops.output import PingOutput, ConnectivityOutput
from ..utils import get_db, SANITIZED_SQLALCHEMY_DATABASE_URI


router = APIRouter()


async def common_response_fields() -> Dict:
    name = get_app_name()
    version = get_app_version()
    pong = datetime.datetime.now().timestamp()
    return {'version': version, 'name': name, 'pong': pong}


@router.get('/ping', summary='Health check', description='Tests to see if service is alive', response_model=PingOutput)
async def ping(common: dict = Depends(common_response_fields)) -> Dict:
    def to_mb(b: int) -> float:
        return b / (1024 * 1024)

    def process_memory_usage_psutil() -> float:
        process = psutil.Process(os.getpid())
        return to_mb(int(process.memory_info().rss))

    def virtual_memory_usage_psutil() -> float:
        return to_mb(int(psutil.virtual_memory().used))

    response = {
        **common,
        **{'virtual_mem_used_MB': virtual_memory_usage_psutil()},
        **{'process_mem_used_MB': process_memory_usage_psutil()}
    }
    logger.info('ping', data={'response': response}, type=LoggingType.REQUEST_RESPONSE_LOG)
    return response


@router.get(
    '/connectivity',
    summary='Check connectivity with dependencies',
    description='Standard endpoint for checking upstream dependencies',
    response_model=ConnectivityOutput
)
def connectivity(common: dict = Depends(common_response_fields), db: Session = Depends(get_db)) -> Dict:
    postgres = {"message": "OK", 'uri': SANITIZED_SQLALCHEMY_DATABASE_URI}
    try:
        rs = db.execute("select 1=1 as connected")
        postgres = {**postgres, **[result for result in rs][0]}
    except Exception as e:
        postgres = {**postgres, **{"message": str(e), "connected": False}}
    response = {**common, **{"dependencies": {"postgres": postgres}}}
    logger.info("connectivity", data={"response": response}, type=LoggingType.REQUEST_RESPONSE_LOG)
    return response

