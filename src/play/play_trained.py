# pylint: disable=no-member
"""
Functionality to visualize trained PPO agents playing Pong games.
The arguments at the end of the file specify which trained
PPO file should be used. This file was not used to generate
any of the data in the results folder.
"""

import argparse
import pygame
from src.algorithms.ppo import PPO
from src.logger.logger import logger
from src.train import constants
from src.pong import constants as pong_constants
from src.pong.game_factory import get_pong_game


class TrainedPlayer:
    """
    Class to make trained agents play Pong
    """

    ARG_MODEL_PATH = "model_path"

    def __init__(self, env_name: str, reward_frequency: str, model_path: str):
        self.env = get_pong_game(env_name)(reward_frequency=reward_frequency)
        self.ppo_agent = PPO(
            constants.STATE_DIMS[env_name],
            constants.ACTION_DIMS[env_name],
            constants.LR_ACTOR,
            constants.LR_CRITIC,
            constants.GAMMA,
            constants.K_EPOCHS,
            constants.EPS_CLIP,
            constants.HAS_CONTINUOUS_ACTION_SPACE,
            constants.ACTION_STD,
            0,
        )

        self.ppo_agent.load(model_path)

    def play(self, num_episodes=5):
        """
        Plays the Pong game specified by the environment name
        """

        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                # Select action
                action = self.ppo_agent.select_action(state)

                # Perform action in the environment
                state, reward, done = self.env.step(action)

                episode_reward += reward

                # Render the game
                self._render_game()
            logger.info("Episode %d Reward: %.6f", episode + 1, episode_reward)

        pygame.quit()

    def _render_game(self, render_wait_time_milliseconds=50):
        """
        Render the game
        """
        self.env.render()
        pygame.time.wait(render_wait_time_milliseconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make trained player play")

    # Add arguments
    parser.add_argument(
        f"--{pong_constants.ARG_ENV_NAME}",
        type=str,
        default=pong_constants.ENV_SIMPLE_PONG,
        help="Name of the environment",
    )
    parser.add_argument(
        f"--{TrainedPlayer.ARG_MODEL_PATH}",
        type=str,
        help="Path to the trained model",
    )

    parser.add_argument(
        f"--{constants.ARG_REWARD_FREQUENCY}",
        type=str,
        default=pong_constants.FREQUENCY_FREQUENT,
        help="Reward frequency",
    )

    args = parser.parse_args()
    trained_player = TrainedPlayer(
        env_name=args.env_name,
        reward_frequency=args.reward_frequency,
        model_path=f"trained/{args.env_name}/{args.model_path}",
    )
    trained_player.play()
