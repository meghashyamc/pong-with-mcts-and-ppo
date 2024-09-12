# pylint: disable=invalid-name
"""
Pong state related functionality needed by MCTS library
"""

from typing import List
from src.train import constants
from src.pong.game_factory import get_pong_game
from src.algorithms.mcts_wrapper import MCTS


class MCTSPongState(MCTS.MCTSState):
    """
    This class exposes Pong state details in the format needed by https://github.com/pbsinclair42/MCTS
    """

    def __init__(self, env_name: str, reward_frequency: str, headless: bool = False):
        self.env_name = env_name
        self.game = get_pong_game(env_name)(
            reward_frequency=reward_frequency, headless=headless
        )
        self.state = self.game.reset()
        self.done = False
        self.reward = 0

    def getCurrentPlayer(self) -> int:
        """
        Get number representing current player
        """
        return 1  # Pong is a single-player game from the AI's perspective

    def getPossibleActions(self):
        """
        Gets list of possible actions
        """
        return list(
            range(
                constants.ACTION_DIMS[self.env_name],
            )
        )

    def takeAction(self, action) -> "MCTSPongState":
        """
        Takes action and returns next state, reward and whether we're done
        """
        state, reward, done = self.game.step(action)
        self.state = state
        self.done = done
        self.reward = reward
        return self

    def isTerminal(self):
        """
        Whether the game is done
        """
        return self.done

    def getReward(self):
        """
        Get reward
        """
        return self.reward
