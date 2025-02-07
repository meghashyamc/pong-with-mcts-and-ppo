# pylint: disable=missing-class-docstring
"""
Models related to Pong games
"""
from enum import Enum
from typing import Tuple
from pydantic import BaseModel


class Velocity(BaseModel):

    x: float
    y: float


class Direction(Enum):
    LEFT = -1
    RIGHT = 1
    STAYPUT = 0


Color = Tuple[int, int, int]
