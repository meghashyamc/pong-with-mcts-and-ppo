"""
Game factory to get the Pong game based on the environment name
"""

from src.pong.base_game import BasePongGame
from src.pong.simple_pong_game import SimplePongGame
from src.pong.complex_pong_game import ComplexPongGame
from src.pong import constants


def get_pong_game(env_name: str) -> BasePongGame:
    """
    Get the Pong game based on the environment name
    """
    match (env_name):
        case constants.ENV_SIMPLE_PONG:
            return SimplePongGame
        case constants.ENV_COMPLEX_PONG:
            return ComplexPongGame
        case _:
            raise ValueError(f"Invalid environment name: {env_name}")
