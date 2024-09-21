# pylint: disable=invalid-name
"""
MCTS related core functionality
"""

import time
from abc import ABC, abstractmethod
from mcts import mcts, randomPolicy
from src.pong.base_game import BasePongGame


class MCTS:
    """
    MCTS wrapper class that internally accesses
    https://github.com/pbsinclair42/MCTS
    """

    def __init__(self, iteration_limit=None, time_limit=None, heuristic: bool = False):
        print("heuristic------------------->", heuristic)
        self.searcher = mcts(
            iterationLimit=iteration_limit,
            timeLimit=time_limit,
            explorationConstant=0.3,
            rolloutPolicy=MCTS.heuristic_rollout if heuristic else randomPolicy,
        )

    @staticmethod
    def heuristic_rollout(state: "MCTS.MCTSState", max_depth=None):
        """
        Rollout a state using a heuristic policy
        """
        depth = 0
        while not state.isTerminal():
            if max_depth and depth >= max_depth:
                break
            depth += 1
            chosen_action = state.getGame().get_action_based_on_heuristic(
                randomness=0.1
            )
            state = state.takeAction(chosen_action)
        return state.getReward()

    class MCTSState(ABC):
        """
        This class exposes state details in the format needed by https://github.com/pbsinclair42/MCTS
        """

        @abstractmethod
        def getGame(self) -> BasePongGame:
            """
            Get the game instance
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
