from logging import Logger, getLogger
from logging.config import dictConfig

CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "format": (
                "%(asctime)s %(levelname)s %(name)s "
                "%(module)s %(funcName)s %(lineno)d "
                "%(process)d %(thread)d %(message)s"
            ),
        },
        "detailed": {
            "format": (
                "%(asctime)s [%(levelname)s] %(name)s "
                "(%(module)s:%(lineno)d) %(funcName)s | %(message)s"
            )
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "stream": "ext://sys.stdout",
        },
        "error_console": {
            "class": "logging.StreamHandler",
            "level": "ERROR",
            "formatter": "detailed",
            "stream": "ext://sys.stderr",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "json",
            "filename": "logs/app.log",
            "maxBytes": 50 * 1024 * 1024,
            "backupCount": 5,
            "encoding": "utf-8",
        },
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console", "error_console", "file"],
    },
}


dictConfig(CONFIG)


def get_logger(name: str) -> Logger:
    return getLogger(name)
