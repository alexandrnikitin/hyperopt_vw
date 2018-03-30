# -*- coding: utf-8 -*-

"""Top-level package for hyperopt vw."""

__author__ = """Alexandr Nikitin"""
__email__ = 'nikitin.alexandr.a@gmail.com'
__version__ = '0.1.0'

import logging
from .search import search
from .objective import Objective

logging.getLogger(__name__).addHandler(logging.NullHandler())
