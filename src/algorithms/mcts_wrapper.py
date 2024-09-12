# pylint: disable=invalid-name
"""
MCTS related core functionality
"""

from typing import Callable
from abc import ABC, abstractmethod
from mcts import mcts


class MCTS:
    """
    MCTS wrapper class that internally accesses
    https://github.com/pbsinclair42/MCTS
    """

    def __init__(self, iteration_limit: int = 100):
        self.searcher = mcts(iterationLimit=iteration_limit)

    class MCTSState(ABC):
        """
        This class exposes state details in the format needed by https://github.com/pbsinclair42/MCTS
        """

        @abstractmethod
        def getCurrentPlayer(self) -> int:
            """
            Get number representing current player
            """

        @abstractmethod
        def getPossibleActions(self):
            """
            Gets list of possible actions
            """

        @abstractmethod
        def takeAction(self, action) -> "MCTSState":
            """
            Takes action and returns next state, reward and whether we're done
            """

        @abstractmethod
        def isTerminal(self):
            """
            Whether the game is done
            """

        @abstractmethod
        def getReward(self):
            """
            Get reward
            """
