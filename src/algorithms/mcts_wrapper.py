# pylint: disable=invalid-name
"""
MCTS wrapper functionality that connects to the core MCTS class.
This is the class accessed during gameplay.
"""

import random
import math
from abc import ABC, abstractmethod

from src.algorithms.mcts import mcts
from src.pong.base_game import BasePongGame
from src.logger.logger import logger
from src.algorithms.ppo import PPO


class MCTS:
    """
    MCTS wrapper class that is called when MCTS functionality needs to be accessed
    during gameplay. It internally accesses core MCTS functionality.
    """

    def __init__(
        self,
        iteration_limit=100,
        time_limit=None,
        ppo_agent: PPO = None,
    ):
        self.ppo_agent = ppo_agent
        rolloutPolicy = self.random_rollout
        if ppo_agent:
            rolloutPolicy = self.random_rollout
            logger.info("using random rollout policy with ppo")

        self.searcher = mcts(
            iterationLimit=iteration_limit,
            timeLimit=time_limit,
            explorationConstant=math.sqrt(2),
            rolloutPolicy=rolloutPolicy,
            ppo_agent=ppo_agent,
        )

    def random_rollout(self, state: "MCTS.MCTSState"):
        """
        Rollout a state using a random policy
        """
        while not state.isTerminal():

            chosen_action = random.choice(state.getPossibleActions())
            state = state.takeAction(chosen_action)

        return state.getReward()

    class MCTSState(ABC):
        """
        This class exposes state details in the format needed by https://github.com/pbsinclair42/MCTS.
        It is essentially an interface that is needed to be accessed by the core MCTS functionality.
        In the case of Pong games, pong/mcts_pong_state.py contains an implementation of this interface.
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

        @abstractmethod
        def getState(self):
            """
            Get the current state of the game.
            """
