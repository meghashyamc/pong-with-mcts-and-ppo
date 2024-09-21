"""
Functionality to visualize MCTS agent playing Pong games
"""

import os
import csv
import random
import time
import argparse
from typing import List, Tuple

import pygame
import numpy as np
from src.train import constants
from src.pong import constants as pong_constants
from src.algorithms.mcts_wrapper import MCTS
from src.pong.mcts_pong_state import MCTSPongState
from src.pong.game_factory import get_pong_game
from src.logger.logger import logger


class RandomPlayer:
    """
    Class to make a agent play Pong with random actions
    """

    def __init__(
        self,
        env_name: str,
        reward_frequency: str,
        render_game: bool = False,
    ):
        self.show_game = render_game
        self.env_name = env_name
        self.env = get_pong_game(env_name)(
            headless=True, reward_frequency=reward_frequency
        )
        self.start_time = time.time()
        self.reward_frequency = reward_frequency
        self._setup_result_paths()

    def _setup_result_paths(self):
        self.csv_folder = f"results/{self.env_name}/random"
        os.makedirs(self.csv_folder, exist_ok=True)
        self.avg_rewards_file = os.path.join(
            self.csv_folder,
            f"avg_rewards_{self.reward_frequency}_{self.start_time}.csv",
        )
        self.avg_paddle_hits_file = os.path.join(
            self.csv_folder,
            f"avg_paddle_hits_{self.reward_frequency}{self.start_time}.csv",
        )
        self.episodes_vs_time_file = os.path.join(
            self.csv_folder,
            f"episodes_vs_time_{self.reward_frequency}_{self.start_time}.csv",
        )

    def play(self, num_episodes=300):
        """
        Plays the Pong game specified by the environment name
        """
        rewards: List[float] = []
        avg_rewards = []
        paddle_hits_history = []
        avg_paddle_hits_list = []
        episodes_vs_time = []

        for episode in range(num_episodes):

            episode_reward = 0
            done = False
            timesteps = 0
            paddle_hits = 0
            possible_actions = list(
                range(
                    constants.ACTION_DIMS[self.env_name],
                )
            )

            while not done:
                action = random.choice(possible_actions)

                state, reward, done = self.env.step(action)

                episode_reward += reward
                timesteps += 1
                if reward == pong_constants.PADDLE_HIT_REWARD:
                    paddle_hits += 1
                # Render the game
                if self.show_game:
                    self._render_game()

            rewards.append(episode_reward)
            paddle_hits_history.append(paddle_hits)
            episodes_vs_time.append((episode, time.time() - self.start_time))

            logger.info(
                f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}"
            )
            logger.info(f"This episode lasted {timesteps} timesteps")
            logger.info(f"This episode had {paddle_hits} paddle hits")

            avg_paddle_hits = (
                np.mean(paddle_hits_history[-constants.WINDOW_SIZE :])
                if len(paddle_hits_history) >= constants.WINDOW_SIZE
                else 0
            )
            if avg_paddle_hits:
                logger.info(
                    f"Average paddle hits for the last {constants.WINDOW_SIZE} episodes: {avg_paddle_hits}"
                )
            avg_paddle_hits_list.append(avg_paddle_hits)

            avg_reward = (
                np.mean(rewards[-constants.WINDOW_SIZE :])
                if len(rewards) >= constants.WINDOW_SIZE
                else 0
            )
            if avg_reward:
                logger.info(
                    f"Average reward for the last {constants.WINDOW_SIZE} episodes: {avg_reward}"
                )
            avg_rewards.append(avg_reward)

        self._save_csv_data(avg_rewards, avg_paddle_hits_list, episodes_vs_time)

    def _save_csv_data(
        self,
        avg_rewards: List[float],
        avg_paddle_hits: List[float],
        episodes_vs_time: List[Tuple[int, float]],
    ):
        with open(self.avg_rewards_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([constants.CSV_COLUMN_EPISODE, constants.CSV_COLUMN_REWARD])
            writer.writerows(enumerate(avg_rewards, 1))

        with open(self.avg_paddle_hits_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [constants.CSV_COLUMN_EPISODE, constants.CSV_COLUMN_PADDLE_HITS]
            )
            writer.writerows(enumerate(avg_paddle_hits, 1))

        with open(self.episodes_vs_time_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([constants.CSV_COLUMN_EPISODE, constants.CSV_COLUMN_TIME])
            writer.writerows(episodes_vs_time)

        logger.info(f"CSV data saved in {self.csv_folder}")

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
        f"--{constants.ARG_REWARD_FREQUENCY}",
        type=str,
        default=pong_constants.FREQUENCY_FREQUENT,
        help="Reward frequency",
    )

    parser.add_argument(
        f"--{constants.ARG_RENDER}",
        type=str,
        default=False,
        help="Render game",
    )

    args = parser.parse_args()
    random_player = RandomPlayer(
        env_name=args.env_name,
        reward_frequency=args.reward_frequency,
        render_game=args.render,
    )
    random_player.play()
