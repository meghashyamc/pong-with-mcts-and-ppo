"""
Train PPO agent for Simple/Complex Pong environment
"""

import os
import csv
import time
import argparse
from typing import Tuple, List
import pygame
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from src.pong.game_factory import get_pong_game
from src.pong import constants as pong_constants
from src.algorithms.PPO import PPO
from src.logger.logger import logger
from src.train import constants

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Device set to: {device}")


class PPOTrainer:
    """
    The class to train the agent for the Simple Pong environment
    """

    def __init__(
        self,
        env_name: str,
        reward_frequency: str,
        show_game_during_training: bool = False,
    ):
        # if headless = True, the game will be not be rendered
        self.env = get_pong_game(env_name)(
            headless=(not show_game_during_training), reward_frequency=reward_frequency
        )
        self.reward_frequency = reward_frequency
        self.env_name = env_name
        self.start_time = time.time()
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
        )

        self._setup_result_paths()

    def train(
        self,
        show_visual_plot_during_training: bool = False,
        show_game_during_training: bool = False,
    ):
        """
        The function to train the PPO agent for the Simple Pong environment
        """

        # track total training time
        logger.info("Started training at (GMT) : %s", self.start_time)

        episodes = []
        rewards = []
        avg_rewards = []
        paddle_hits_history = []
        avg_paddle_hits_list = []
        episodes_vs_time = []
        i_episode = 0
        fig, axes, line1, line2 = None, None, None, None
        if show_visual_plot_during_training:
            fig, axes, line1, line2 = self._setup_plot()

        for i_episode in range(constants.MAX_EPISODES):

            state = self.env.reset()
            episode_reward = 0
            time_step = 0
            paddle_hits = 0

            while True:
                # select action with policy
                action = self.ppo_agent.select_action(state)

                # Step the environment
                next_state, reward, done = self.env.step(action)
                if reward == pong_constants.PADDLE_HIT_REWARD:
                    paddle_hits += 1
                # Render the game
                if show_game_during_training:
                    self._render_game()
                # Saving reward and is_terminals
                self.ppo_agent.buffer.rewards.append(reward)
                self.ppo_agent.buffer.is_terminals.append(done)

                time_step += 1
                episode_reward += reward
                state = next_state

                # Break if the episode is over
                if done:
                    break

            # Update every episode
            self.ppo_agent.update()

            logger.info("Reward for episode %d was %d", i_episode, episode_reward)
            logger.info("This episode lasted %d timesteps", time_step)
            logger.info("This episode had %d paddle hits", paddle_hits)
            episodes.append(i_episode)
            rewards.append(episode_reward)
            paddle_hits_history.append(paddle_hits)

            self._log_average_reward(
                episodes, rewards, avg_rewards, show_visual_plot_during_training, line1
            )
            episodes_vs_time.append((i_episode, time.time() - self.start_time))
            enviroment_solved = self._log_average_paddle_hits(
                episodes,
                paddle_hits_history,
                avg_paddle_hits_list,
                show_visual_plot_during_training,
                line2,
            )
            if enviroment_solved:
                logger.info("Environment solved!")
                break
            if show_visual_plot_during_training:
                self._draw_plot(fig, axes)

        self._save_model()
        self._save_csv_data(avg_rewards, avg_paddle_hits_list, episodes_vs_time)

        # Print total training time
        logger.info(f"Total training time: {time.time() - self.start_time}")

        self._continue_showing_plot()

    def _setup_result_paths(self):
        self.csv_folder = f"results/{self.env_name}"
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

    def _save_model(self):
        final_model_path = os.path.join(
            constants.TRAINED_FOLDER_NAME,
            self.env_name,
            f"PPO_{self.reward_frequency}_{time.time()}.pth",
        )
        self.ppo_agent.save(final_model_path)
        logger.info(f"Final model saved at: {final_model_path}")

    def _setup_plot(self) -> Tuple[Figure, Axes, Line2D, Line2D]:
        """
        Sets up basic plot to show UI during training
        """
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 5))
        (line1,) = ax.plot([], [], "b-", label="Avg Reward")
        (line2,) = ax.plot([], [], "r-", label="Avg Paddle Hits")

        ax.set_xlabel("Episode")
        ax.set_title("Training Progress")
        ax.legend()

        plt.show()
        return fig, ax, line1, line2

    def _log_average_paddle_hits(
        self,
        episodes: List[int],
        paddle_hits_history: List[int],
        avg_paddle_hits_list: List[int],
        show_visual_plot_during_training: bool,
        line: Line2D,
    ) -> bool:
        """
        Logs average paddle hits and shows the plot if needed.
        Returns True if the environment is solved.
        """
        avg_paddle_hits = (
            np.mean(paddle_hits_history[-constants.WINDOW_SIZE :])
            if len(paddle_hits_history) >= constants.WINDOW_SIZE
            else 0
        )
        avg_paddle_hits_list.append(avg_paddle_hits)
        if show_visual_plot_during_training:
            line.set_xdata(episodes)
            line.set_ydata(avg_paddle_hits_list)

        logger.info(
            "Average paddle hits over last %d episodes: %.2f",
            constants.WINDOW_SIZE,
            avg_paddle_hits,
        )
        return avg_paddle_hits >= constants.PADDLE_HITS_THRESHOLD

    def _draw_plot(self, fig, ax):

        plot_pause = 0.001  # seconds
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(plot_pause)

    def _continue_showing_plot(self):
        """
        Continues showing the plot for average reward and average paddle hits during training
        even after the training is done
        """
        plt.ioff()
        plt.show()

    def _render_game(self, render_wait_time_milliseconds=20):
        """
        Render the game
        """
        self.env.render()
        pygame.time.wait(render_wait_time_milliseconds)

    def _log_average_reward(
        self,
        episodes: List[int],
        rewards: List[float],
        avg_rewards: List[float],
        show_visual_plot_during_training: bool,
        line: Line2D,
    ) -> float:
        """
        Log the average reward and show the plot if needed
        """
        avg_reward = (
            np.mean(rewards[-constants.WINDOW_SIZE :])
            if len(rewards) >= constants.WINDOW_SIZE
            else 0
        )
        avg_rewards.append(avg_reward)
        if show_visual_plot_during_training:
            line.set_xdata(episodes)
            line.set_ydata(avg_rewards)
        logger.info(
            "Average reward over last %d episodes: %d",
            constants.WINDOW_SIZE,
            avg_reward,
        )
        return float(avg_reward)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Trainer")

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

    args = parser.parse_args()
    ppo_trainer = PPOTrainer(
        env_name=args.env_name, reward_frequency=args.reward_frequency
    )
    ppo_trainer.train()
