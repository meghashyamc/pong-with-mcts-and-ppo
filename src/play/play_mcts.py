"""
Functionality to record or visualize MCTS agent playing Pong games.
This file should be run with the appropriate arguments mentioned towards the end of this
file to make the agent:
- Play Simple/Complex Pong using  only MCTS
- Play Simple/Complex Pong using MCTS aided by pre-trained PPO. n this case,
  the pre-trained PPO file (.pth) should be present
  at the path PPO_MODEL_TRAINED_SIMPLE_PONG/PPO_MODEL_TRAINED_COMPLEX_PONG
  in the 'trained' folder.
This file was used to make MCTS and MCTS aided by pre-trained PPO play Pong games
to generate the data in results/ComplexPong/mcts and results/SimplePong/mcts.
"""

import os
import csv
import time
import argparse
from typing import List, Tuple, Optional

import pygame
import numpy as np
from src.train import constants
from src.pong import constants as pong_constants
from src.algorithms.mcts_wrapper import MCTS
from src.pong.mcts_pong_state import MCTSPongState
from src.pong.game_factory import get_pong_game
from src.logger.logger import logger
from src.algorithms.ppo import PPO
from src.pong.constants import ENV_SIMPLE_PONG, ENV_COMPLEX_PONG


class MCTSPlayer:
    """
    Class to make an MCTS agent play Pong
    """

    ARG_NUM_OF_ITERATIONS = "iterations"
    ARG_TIME_LIMIT = "timelimit"
    PPO_MODEL_TRAINED_SIMPLE_PONG = "PPO_frequent_1725170148.2600577.pth"
    PPO_MODEL_TRAINED_COMPLEX_PONG = "PPO_frequent_1725725991.5131817.pth"

    def __init__(
        self,
        env_name: str,
        reward_frequency: str,
        num_of_iterations: Optional[int],
        time_limit: Optional[int],
        render_game: bool = False,
        ppo: bool = False,
    ):
        self.show_game = render_game
        self.env_name = env_name
        self.env = get_pong_game(env_name)(
            headless=(not self.show_game), reward_frequency=reward_frequency
        )
        self.start_time = time.time()
        self.num_of_iterations = num_of_iterations
        self.time_limit = time_limit
        self.ppo = ppo
        self.ppo_agent = None
        if self.ppo:
            self.ppo_agent = self._get_ppo_agent()
        self.mcts_agent = MCTS(
            iteration_limit=self.num_of_iterations,
            time_limit=self.time_limit,
            ppo_agent=self.ppo_agent,
        )
        self.reward_frequency = reward_frequency

        self._setup_result_paths()

    def _get_ppo_agent(self) -> PPO:
        ppo_file = ""
        if self.env_name == ENV_SIMPLE_PONG:
            ppo_file = self.PPO_MODEL_TRAINED_SIMPLE_PONG
        elif self.env_name == ENV_COMPLEX_PONG:
            ppo_file = self.PPO_MODEL_TRAINED_COMPLEX_PONG
        else:
            raise ValueError(f"Invalid environment name: {self.env_name}")

        model_path = f"trained/{self.env_name}/{ppo_file}"
        ppo_agent = PPO(
            constants.STATE_DIMS[self.env_name],
            constants.ACTION_DIMS[self.env_name],
            0,
            0,
            constants.GAMMA,
            0,
            constants.EPS_CLIP,
            constants.HAS_CONTINUOUS_ACTION_SPACE,
            constants.ACTION_STD,
            0,
        )
        ppo_agent.load(model_path)
        return ppo_agent

    def _setup_result_paths(self):

        if self.ppo:
            self.csv_folder = f"results/{self.env_name}/mcts/ppo"
        else:
            self.csv_folder = f"results/{self.env_name}/mcts/original"
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
            self.env.reset()

            while not done:
                initial_game_state = self.env.clone()
                mcts_pong_state = MCTSPongState(
                    self.env_name,
                    initial_game_state,
                    reward_frequency=self.reward_frequency,
                )
                action = self.mcts_agent.searcher.search(initialState=mcts_pong_state)
                logger.debug(f"chose action: {action}")
                _, reward, done = self.env.step(action)
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
        f"--{MCTSPlayer.ARG_NUM_OF_ITERATIONS}",
        type=int,
        default=None,
        help="Max number of iterations per action",
    )

    parser.add_argument(
        f"--{MCTSPlayer.ARG_TIME_LIMIT}",
        type=int,
        default=None,
        help="Time limit per move for MCTS in ms",
    )

    parser.add_argument(
        f"--{constants.ARG_RENDER}",
        type=bool,
        default=False,
        help="Render game",
    )

    parser.add_argument(
        f"--{constants.ARG_PPO}",
        type=bool,
        default=False,
        help="Use ppo",
    )

    args = parser.parse_args()
    mcts_player = MCTSPlayer(
        env_name=args.env_name,
        reward_frequency=args.reward_frequency,
        num_of_iterations=args.iterations,
        time_limit=args.timelimit,
        render_game=args.render,
        ppo=args.ppo,
    )
    mcts_player.play()
