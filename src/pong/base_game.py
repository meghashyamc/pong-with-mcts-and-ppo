"""
Common methods implemented by all Pong games
"""

from abc import ABC, abstractmethod
from typing import List, Tuple


class BasePongGame(ABC):
    """
    Interface implemented by all Pong games
    """

    @abstractmethod
    def reset(self):
        """Reset the game state and return the initial state."""

    @abstractmethod
    def step(self, action: int) -> Tuple[List, float, bool]:
        """Take a step in the environment. A step involves taking an action,
        updating the game state and returning the next state, reward and whether
        the episode is done"""

    @abstractmethod
    def update(self):
        """Update game state."""

    @abstractmethod
    def update_reward(self, reward: float):
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
