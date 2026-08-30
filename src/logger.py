import json
import logging
import sys
from datetime import datetime


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "time": datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
            "level": record.levelname,
            "module": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)

    try:
        from src.config import settings
        log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    except Exception:
        log_level = logging.INFO

    logger.setLevel(log_level)
    return logger