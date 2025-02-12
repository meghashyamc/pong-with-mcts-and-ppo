"""
This file contains functionality to train a PPO/PPO-MCTS agent
to play Simple/Complex Pong. Running this file will start the training based
on the arguments passed (as specified towards the end of the file). After the
training is complete, trained '.pth' file will be saved in the 'trained' folder.
"""

import os
import csv
import time
import random
import argparse
from collections import deque
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
from src.algorithms.ppo import PPO
from src.logger.logger import logger
from src.train import constants
from src.algorithms.mcts_wrapper import MCTS
from src.pong.mcts_pong_state import MCTSPongState


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Device set to: {device}")


class PPOTrainer:
    """
    The class to train the agent for the Simple Pong/Complex Pong environments
    """

    def __init__(
        self,
        env_name: str,
        reward_frequency: str,
        show_game_during_training: bool = False,
        show_visual_plot_during_training: bool = False,
        use_mcts: bool = False,
    ):
        self.show_game_during_training = show_game_during_training
        self.show_visual_plot_during_training = show_visual_plot_during_training
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
            constants.IMITATION_LOSS_WEIGHTS[env_name],
        )

        self.use_mcts = use_mcts
        if use_mcts:
            self.mcts_agent = MCTS()
        self.ppo_choice_history = deque(maxlen=100)
        logger.info(f"using mcts: {self.use_mcts}")
        self._setup_result_paths()

    def train(
        self,
    ):
        """
        The main function to train the PPO/PPO-MCTS agent to play Simple/Complex Pong.
        """
        show_game_during_training: bool = self.show_game_during_training
        show_visual_plot_during_training: bool = self.show_visual_plot_during_training
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

                action = self.select_action(i_episode, state)

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
            logger.info(
                "Recent PPO 'is better' percentage: %.2f",
                self._get_ppo_is_better_percentage(),
            )

            logger.info("Reward for episode %d was %d", i_episode, episode_reward)
            logger.info("This episode lasted %d timesteps", time_step)
            logger.info("This episode had %d paddle hits", paddle_hits)
            episodes.append(i_episode)
            rewards.append(episode_reward)
            paddle_hits_history.append(paddle_hits)

            # self._update_mcts_probability(rewards)
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

    def select_action(
        self,
        episode: int,
        state: List[float],
    ) -> int:
        """
        Select action to take - either using MCTS or PPO.
        """
        ppo_action = None
        mcts_action = constants.DUMMY_ACTION
        mcts_action_tensor = torch.tensor(mcts_action).to(device)
        is_mcts_better = False

        if not self.use_mcts:
            ppo_action = self.ppo_agent.select_action(state)
            self._update_action_choice(mcts_action, is_mcts_better)
            return ppo_action

        # Calculate the percentage of PPO choices in the last 100 decisions
        ppo_percentage = self._get_ppo_is_better_percentage()
        if (
            ppo_percentage >= self._get_ppo_percentage_threshold(episode)
            and len(self.ppo_choice_history) == constants.LENGTH_PPO_CHOICE_HISTORY
        ):
            ppo_action = self.ppo_agent.select_action(state)
            self._update_action_choice(mcts_action, is_mcts_better)
            return ppo_action

        simulated_game_ppo = self.env.clone()

        with torch.no_grad():
            ppo_state = torch.FloatTensor(state).to(device)
            ppo_action_result, _, _ = self.ppo_agent.policy_old.act(ppo_state)
            ppo_action = ppo_action_result.item()

        _, ppo_reward, _ = simulated_game_ppo.step(ppo_action)
        if ppo_reward == pong_constants.PADDLE_HIT_REWARD:
            ppo_action = self.ppo_agent.select_action(state)
            self._update_action_choice(mcts_action, is_mcts_better)
            return ppo_action

        simulated_game_mcts = self.env.clone()
        mcts_pong_state = MCTSPongState(
            self.env_name,
            simulated_game_mcts,
            reward_frequency=self.reward_frequency,
        )
        simulated_mcts_action = self.mcts_agent.searcher.search(
            initialState=mcts_pong_state
        )
        _, mcts_reward, _ = simulated_game_mcts.step(simulated_mcts_action)

        if (ppo_reward >= mcts_reward) and (ppo_reward > 0):
            ppo_action = self.ppo_agent.select_action(state)
            self._update_action_choice(mcts_action, is_mcts_better)
            return ppo_action

        if (ppo_reward == mcts_reward) and (ppo_reward == 0):
            ppo_chosen = random.choice([True, False])
            if ppo_chosen:
                ppo_action = self.ppo_agent.select_action(state)
                self._update_action_choice(mcts_action, is_mcts_better)
                return ppo_action

        mcts_action = simulated_mcts_action
        is_mcts_better = True
        with torch.no_grad():
            mcts_action_tensor = torch.tensor(mcts_action).to(device)
            if not self.ppo_agent.has_continuous_action_space:
                mcts_action_tensor = mcts_action_tensor.long()

            state_tensor = torch.FloatTensor(state).to(device)
            mcts_action_logprob, state_value, _ = self.ppo_agent.policy_old.evaluate(
                state_tensor, mcts_action_tensor
            )
            mcts_action_logprob = mcts_action_logprob.detach()
            state_value = state_value.detach()

        # Store data in PPO buffer
        self.ppo_agent.buffer.states.append(state_tensor)
        self.ppo_agent.buffer.actions.append(mcts_action_tensor)
        self.ppo_agent.buffer.logprobs.append(mcts_action_logprob)
        self.ppo_agent.buffer.state_values.append(state_value)

        self._update_action_choice(mcts_action, is_mcts_better)
        return mcts_action

    def _get_ppo_percentage_threshold(self, episode: int) -> float:
        if self.env_name == pong_constants.ENV_SIMPLE_PONG:
            return max(0, 1 - (episode / 500))
        if self.env_name == pong_constants.ENV_COMPLEX_PONG:
            return max(0, 1 - (episode / 1000))
        raise Exception(
            "Invalid environment name when calculating PPO percentage threshold"
        )

    def _update_action_choice(self, mcts_action: int, is_mcts_better: bool):
        self.ppo_agent.buffer.mcts_actions.append(mcts_action)
        self.ppo_agent.buffer.is_mcts_better.append(is_mcts_better)
        self.ppo_choice_history.append(1 if (not is_mcts_better) else 0)

    def _get_ppo_is_better_percentage(self) -> float:
        if not self.use_mcts:
            return 1

        # Calculate the percentage of PPO choices in the last 100 decisions
        ppo_percentage = (
            sum(self.ppo_choice_history) / len(self.ppo_choice_history)
            if self.ppo_choice_history
            else 0
        )
        return ppo_percentage

    def _setup_result_paths(self):
        if not self.use_mcts:
            self.csv_folder = f"results/{self.env_name}/ppo"
        else:
            self.csv_folder = f"results/{self.env_name}/ppo-mcts/original"  # Original just means we are using 'pure' MCTS
        os.makedirs(self.csv_folder, exist_ok=True)
        self.avg_rewards_file = os.path.join(
            self.csv_folder,
            f"avg_rewards_{self.reward_frequency}_{self.start_time}.csv",
        )
        self.avg_paddle_hits_file = os.path.join(
            self.csv_folder,
            f"avg_paddle_hits_{self.reward_frequency}_{self.start_time}.csv",
        )
        self.episodes_vs_time_file = os.path.join(
            self.csv_folder,
            f"episodes_vs_time_{self.reward_frequency}_{self.start_time}.csv",
        )

    def _save_model(self):
        mcts_path_component = "_mcts" if self.use_mcts else ""
        final_model_path = os.path.join(
            constants.TRAINED_FOLDER_NAME,
            self.env_name,
            f"PPO{mcts_path_component}_{self.reward_frequency}_{time.time()}.pth",
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

    parser.add_argument(
        f"--{constants.ARG_RENDER}",
        type=bool,
        default=False,
        help="Render game during training",
    )

    parser.add_argument(
        f"--{constants.ARG_PLOT}",
        type=bool,
        default=False,
        help="Plot average paddle hits during training",
    )

    parser.add_argument(
        f"--{constants.USE_MCTS}",
        type=bool,
        default=False,
        help="Use MCTS to support PPO",
    )

    args = parser.parse_args()
    ppo_trainer = PPOTrainer(
        env_name=args.env_name,
        reward_frequency=args.reward_frequency,
        show_game_during_training=args.render,
        show_visual_plot_during_training=args.plot,
        use_mcts=args.mcts,
    )
    ppo_trainer.train()
