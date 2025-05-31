import datetime as dt
import json
import logging
import atexit
import logging.config
import logging.handlers
import pathlib
import queue
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

def setup_logging():
    pathlib.Path("logs").mkdir(exist_ok=True)

    simple_formatter = logging.Formatter(
        "%(asctime)s: %(name)s (%(levelname)s) - %(message)s"
    )

    json_formatter = MyJSONFormatter(
        fmt_keys={
            "level": "levelname",
            "logger": "name",
            "module": "module",
            "function": "funcName",
            "line": "lineno",
            "thread_name": "threadName"
        }
    )

    stderr_handler = logging.StreamHandler()
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(simple_formatter)

    file_handler = logging.handlers.RotatingFileHandler(
        "logs/f1_predictor.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=3,
        # encoding="utf-8" # Dont really want this rn
    )

    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(json_formatter)

    matplotlib_filter = MatPlotlibFilter()
    stderr_handler.addFilter(matplotlib_filter)
    file_handler.addFilter(matplotlib_filter)

    log_queue = queue.Queue()
    queue_handler = logging.handlers.QueueHandler(log_queue)

    listener = logging.handlers.QueueListener(
        log_queue,
        stderr_handler,
        file_handler,
    )

    listener.start()

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(queue_handler)

    atexit.register(listener.stop)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

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

class MatPlotlibFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> Union[bool, logging.LogRecord]:
        return not (record.name == 'matplotlib.font_manager' and record.levelno == logging.DEBUG)
