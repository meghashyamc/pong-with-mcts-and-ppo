# pylint: disable=invalid-name
"""
Pong state related functionality needed by MCTS library.
"""

from typing import List
from src.train import constants
from src.pong import constants as pong_constants
from src.pong.base_game import BasePongGame
from src.algorithms.mcts_wrapper import MCTS


class MCTSPongState(MCTS.MCTSState):
    """
    This class exposes Pong state details in the format needed by https://github.com/pbsinclair42/MCTS
    """

    def __init__(
        self,
        env_name: str,
        env: BasePongGame,
        reward_frequency: str,
        game: BasePongGame = None,
        done: bool = False,
        reward: float = 0,
    ):
        self.env_name = env_name
        self.reward_frequency = reward_frequency

        if game:
            # use the provided game (already cloned in takeAction)
            self.game = game
            self.done = done
            self.reward = reward
            return

        # clone the initial game state
        self.game = env
        self.game.reset()
        self.done = False
        self.reward = 0

    def getGame(self) -> BasePongGame:
        """
        Get the game instance
        """
        return self.game

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

    def getStates(self):
        """
        Gets list of possible states
        """
        return self.game

    def takeAction(self, action) -> "MCTSPongState":
        """
        Takes action and returns next state, reward and whether we're done
        """
        new_game = self.game.clone()
        # Apply the action to the cloned game
        _, reward, done = new_game.step(action)
        # Apply the action to the current game state

        # Create a new state
        new_state = MCTSPongState(
            self.env_name,
            env=None,
            reward_frequency=self.reward_frequency,
            game=new_game,
            done=done,
            reward=reward,
        )
        return new_state

    def isTerminal(self):
        """
        Whether the game is done
        """
        return (self.reward == pong_constants.PADDLE_HIT_REWARD) or (
            self.reward == pong_constants.BALL_FALLING_THROUGH_REWARD
        )

    def getReward(self):
        """
        Get reward
        """
        if self.reward == pong_constants.BALL_FALLING_THROUGH_REWARD:
            return 0
        if self.reward == pong_constants.PADDLE_HIT_REWARD:
            return 1
        raise ValueError(f"Invalid reward: {self.reward}")

    def getState(self):
        """
        Get the current state of the game.
        """
        return self.game.get_state()

    def __str__(self):
        return f"{self.game}"
