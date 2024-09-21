# pylint: disable=invalid-name
"""
Pong state related functionality needed by MCTS library
"""

from typing import List
from src.train import constants
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
        heuristic: bool = False,
        game: BasePongGame = None,
        done: bool = False,
        reward: float = 0,
    ):
        self.env_name = env_name
        self.heuristic = heuristic
        self.reward_frequency = reward_frequency

        if game:
            # use the provided game (already cloned in takeAction)
            self.game = game
            self.done = done
            self.reward = reward
            return

        # clone the initial game state
        self.game = env.clone()
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

    def takeAction(self, action) -> "MCTSPongState":
        """
        Takes action and returns next state, reward and whether we're done
        """
        new_game = self.game.clone()
        _, reward, done = new_game.step(action)
        # create a new MCTSPongState with the updated game
        new_state = MCTSPongState(
            self.env_name,
            env=None,  # we use the cloned game, so no need to pass env
            reward_frequency=self.reward_frequency,
            heuristic=self.heuristic,
            game=new_game,
            done=done,
            reward=reward,
        )
        return new_state

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
