import logging
from logging import handlers
from basis import config
from core import utils

log_path = utils.create_directories(config.LOG_PATH)
# logging.basicConfig(filename= log_path / 'main.log', filemode='a',
# 	format='%(asctime)s [%(levelname)s] (%(name)s) %(message)s', level=logging.DEBUG)
# logging.info('Program Start')

global logger
logger = logging.getLogger(config.PROJECT_NAME)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] (%(name)s) %(message)s')
file_handler = handlers.RotatingFileHandler(filename=log_path / '{}.log'.format(config.PROJECT_NAME), mode='a',
	maxBytes=2097152, backupCount=20)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)