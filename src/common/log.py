import sys
from loguru import logger
from src.common.settings import settings


LOGURU_FORMAT = "<level>{level}</level> | " \
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | " \
                "<cyan>{module}</cyan> | " \
                "<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"


def setup_logger():
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        # format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        format=LOGURU_FORMAT
    )
    return logger

logger = setup_logger()
