from langchain_community.document_loaders.pdf import PyPDFLoader
import os
import logging
import logging.config

#Configure logging for error handling
def configure_logging():
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '[%(levelname)s] %(asctime)s  %(message)s',
                'datefmt': '%d-%m-%Y %H:%M:%S',
            },
        },
        'handlers': {
            'file': {
                'level': 'DEBUG',
                'class': 'logging.FileHandler',
                'filename': 'application.log',
                'formatter': 'standard',
            },
            'console': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
            },
        },
        'loggers': {
            '': {
                'handlers': ['file', 'console'],
                'level': 'INFO',
                'propagate': True,
            }
        },
    })
    return

#Activate logger for specific unit
def get_logger(name=None):
    return logging.getLogger(name)

#Get PDF file and load it
def load_pdf_file(path, file):
    loader = PyPDFLoader(os.path.join(path, file))
    return "File loaded."

