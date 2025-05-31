import datetime as dt
import json
import logging
from typing import Union, Dict, Optional

LOG_RECORD_BUILTIN_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


class MyJSONFormatter(logging.Formatter):
    def __init__(
            self,
            *,
            fmt_keys: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        self.fmt_keys = fmt_keys if fmt_keys is not None else {}

    def format(self, record: logging.LogRecord) -> str:
        message = self._prepare_log_dict(record)
        return json.dumps(message, default=str)

    def _prepare_log_dict(self, record: logging.LogRecord) -> dict:
        permanent_fields = {
            "message": record.getMessage(),
            "timestamp": dt.datetime.fromtimestamp(
                record.created, tz=dt.timezone.utc
            ).isoformat(),
        }

        if record.exc_info is not None:
            permanent_fields["exc_info"] = self.formatException(record.exc_info)

        if record.stack_info is not None:
            permanent_fields["stack_info"] = self.formatStack(record.stack_info)

        message = {}
        for key, val in self.fmt_keys.items():
            # Skip timestamp since we handle it in permanent_fields
            if key == "timestamp":
                continue

            msg_val = record.__dict__.get(key, None)
            if msg_val is not None:
                message[key] = msg_val
            else:
                # Handle the mapping from config key to LogRecord attribute
                if hasattr(record, val):
                    message[key] = getattr(record, val)
                else:
                    # If attribute doesn't exist, skip it or set to None
                    message[key] = None

        message.update(permanent_fields)

        for key, val in record.__dict__.items():
            if key not in LOG_RECORD_BUILTIN_ATTRS:
                message[key] = val

        return message


class NonErrorFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> Union[bool, logging.LogRecord]:
        return record.levelno <= logging.ERROR