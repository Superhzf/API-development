# Standard build-in libraries
from collections import OrderedDict
import logging
import sys
from typing import Any, Dict, Union
from logging import LogRecord
from datetime import datetime
import os

# Related third party libraries
import structlog
from pythonjsonlogger import jsonlogger

# Local application/library specific imports
from ..models import EstEnum
from ..utils import get_app_name, get_app_version


logger = structlog.get_logger('uvicorn')


class LoggingType(EstEnum):
    EXCEPTION = 'exception'
    ACCESS_REQUEST = 'access-request'
    REQUEST_RESPONSE_LOG = 'request-response-log'
    STARTUP_MESSAGE = 'startup-message'
    SHUTDOWN_MESSAGE = 'shutdown-message'
    DB_OPS = 'db-ops'
    MISC = 'misc'


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self,
                   log_record: Dict,
                   record: Union[Any, LogRecord],
                   message_dict: Dict) -> None:
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        # date
        if not log_record.get('date'):
            log_record['date'] = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        # Service
        service = {'name': get_app_name(), 'version': get_app_version()}
        log_record['service'] = service
        # type
        if 'exception' in record.getMessage().lower():
            log_record['type'] = LoggingType.EXCEPTION
        elif 'HTTP' in record.getMessage():
            log_record['type'] = LoggingType.ACCESS_REQUEST
            # Logging library does not support typing which make type declarations difficult
            access_request_fields = {
                'client': {
                    'ip': record.args[0][0],  # type: ignore
                    'port': record.args[0][0],  # type: ignore
                },
                'method': record.args[1],  # type: ignore
                'path': record.args[2],  # type: ignore
                'http_version': record.args[3],  # type: ignore
                'status_code': record.args[4]  # type: ignore
            }
            log_record['data'] = {
                **log_record.get('data', {}),
                **access_request_fields,
            }
        elif any(stop_str in record.getMessage().lower() for stop_str in ['start', 'running']):
            log_record['type'] = LoggingType.STARTUP_MESSAGE
        elif any(stop_str in record.getMessage().lower() for stop_str in ['shut', 'finish']):
            log_record['type'] = LoggingType.SHUTDOWN_MESSAGE

        if 'type' not in log_record:
            log_record['type'] = LoggingType.MISC

        # Convert message to data -> event
        common_log_fields = {
            'event': record.getMessage(),
            'path_name': record.pathname,
            'file_name': record.pathname,
            'function_name': record.funcName,
            'line_no': record.lineno,
            'level': record.levelname,
            'logger_name': record.name
        }

        if log_record['type'] == LoggingType.ACCESS_REQUEST:
            del common_log_fields['event']
        log_record['data'] = {
            **log_record.get('data', {}),
            **common_log_fields,
        }

        # tags
        if log_record['type'] == LoggingType.EXCEPTION:
            log_record['data'] = {
                **log_record.get("data", {}),
                **{"exc_info": log_record.get("exc_info", {})},
            }
            log_record['tags'] = ['error', 'exception']
        else:
            log_record['tags'] = ['general']

        # clean up root fields that do not now appear in the data
        if log_record.get("message"):
            del log_record["message"]
        if log_record.get("level"):
            del log_record["level"]
        if log_record.get("logger"):
            del log_record["logger"]
        if log_record.get("exc_info"):
            del log_record["exc_info"]


def logging_text() -> None:
    # This works with colorama and dev console logging
    structlog.configure(
        processors=[
            structlog.stdlib.add_logger_name,
            # Add log level to the event dict: ['INFO','warning','error']
            structlog.stdlib.add_log_level,
            # Apply stdlib-like string formatting to the event key.
            structlog.stdlib.PositionalArgumentsFormatter(),
            # Add a timestamp to event_dict.
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M.%S"),
            # Add stack information
            structlog.processors.StackInfoRenderer(),
            # Replace an exc_info field by an exception string field
            structlog.processors.format_exc_info,
            # Render event_dict nicely aligned, possibly in colors, and ordered.
            structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def logging_json() -> None:
    structlog.configure(
        processors=[
            # https://www.structlog.org/en/stable/standard-library.html?highlight=level#rendering-using-logging-based-formatters
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            # Render event_dict into keyword arguments for logging.log.
            structlog.stdlib.render_to_log_kwargs,
        ],
        context_class=OrderedDict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.BoundLogger,
        cache_logger_on_first_use=True,
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CustomJsonFormatter())
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)


def logging_setup() -> None:
    # logging_style = os.environ.get('logging', 'json')
    # if logging_style == 'text':
    #     logging_text()
    # else:
    logging_json()
