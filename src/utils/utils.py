"""
Common utility functions used by various packages
"""

from src.logger.logger import logger


def print_horizontal_line():
    """
    Print a horizontal line to the console
    """
    logger.info("=" * 84)
