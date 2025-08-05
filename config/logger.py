import logging
import logging.config
import os

os.makedirs("logs", exist_ok=True)

# Define logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{asctime} {levelname} {name} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
        'app': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': 'logs/app.log',
            'formatter': 'verbose',
        },
        'models': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': 'logs/models.log',
            'formatter': 'verbose',
        },
        'etl': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': 'logs/etl.log',
            'formatter': 'verbose',
        },
    },
    'loggers': {
        '': {
            'handlers': ['console', 'app'],
            'level': 'DEBUG',
            'propagate': True,
        },
        'models': {
            'handlers': ['models'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'etl': {
            'handlers': ['etl'],
            'level': 'DEBUG',
            'propagate': False,
        },
    },
}

def setup_logging() -> None:
    logging.config.dictConfig(LOGGING_CONFIG)