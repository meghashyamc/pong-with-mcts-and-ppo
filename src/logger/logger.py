"""
Logging module for the project
"""

import logging

# Set up logging
# Change logging level to DEBUG for more detailed logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
