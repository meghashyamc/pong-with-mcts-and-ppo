"""
Common methods implemented by all Pong games
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
from src.pong.simple_pong_game import SimplePongGame
from src.pong.complex_pong_game import ComplexPongGame
from src.pong import constants


class BasePongGame(ABC):
    """
    Interface implemented by all Pong games
    """

    @abstractmethod
    def reset(self):
        """Reset the game state and return the initial state."""

    @abstractmethod
    def step(self, action) -> Tuple[List, float, bool]:
        """Take a step in the environment. A step involves taking an action,
        updating the game state and returning the next state, reward and whether
        the episode is done"""

    @abstractmethod
    def update(self):
        """Update game state."""

    @abstractmethod
    def update_reward(self):
        """Update the reward based on the reward received or the default reward."""

    @abstractmethod
    def get_default_reward(self) -> float:
        """Get the default reward for an action."""

    @abstractmethod
    def render(self):
        """Render the current game state."""

    @abstractmethod
    def close(self):
        """Close the Pygame window."""

    @abstractmethod
    def run(self):
        """Main game loop for human play"""

    @staticmethod
    def get_pong_game(env_name: str) -> "BasePongGame":
        """
        Get the Pong game based on the environment name
        """
        match (env_name):
            case constants.ENV_SIMPLE_PONG:
                return SimplePongGame()
            case constants.ENV_COMPLEX_PONG:
                return ComplexPongGame()
            case _:
                raise ValueError(f"Invalid environment name: {env_name}")
