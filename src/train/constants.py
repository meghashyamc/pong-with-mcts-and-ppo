"""
Constants related to training PPO/PPO-MCTS.
"""

from src.pong import constants

ARG_REWARD_FREQUENCY = "reward_frequency"
ARG_RENDER = "render"
ARG_PPO = "ppo"
ARG_PLOT = "plot"

USE_MCTS = "mcts"

TRAINED_FOLDER_NAME = "trained"
CSV_COLUMN_EPISODE = "Episode"
CSV_COLUMN_TIME = "Time (seconds)"
CSV_COLUMN_REWARD = "Average Reward"
CSV_COLUMN_PADDLE_HITS = "Average Paddle Hits"

IMITATION_LOSS_WEIGHT_SIMPLE_PONG = 0.1
IMITATION_LOSS_WEIGHT_COMPLEX_PONG = 0.03

STATE_DIMS = {
    constants.ENV_SIMPLE_PONG: 5,
    constants.ENV_COMPLEX_PONG: 7,
}

ACTION_DIMS = {
    constants.ENV_SIMPLE_PONG: 3,
    constants.ENV_COMPLEX_PONG: 9,
}

IMITATION_LOSS_WEIGHTS = {
    constants.ENV_SIMPLE_PONG: IMITATION_LOSS_WEIGHT_SIMPLE_PONG,
    constants.ENV_COMPLEX_PONG: IMITATION_LOSS_WEIGHT_COMPLEX_PONG,
}


HAS_CONTINUOUS_ACTION_SPACE = False  # discrete action space
MAX_EPISODES = int(50000)
ACTION_STD = None  # constant std for action distribution (Multivariate Normal)
K_EPOCHS = 4  # update policy for K epochs
EPS_CLIP = 0.2  # clip parameter for PPO
GAMMA = 0.99  # discount factor
LR_ACTOR = 0.0003  # learning rate for actor network
LR_CRITIC = 0.0003  # learning rate for critic network
WINDOW_SIZE = 100  # Number of episodes to average over for various metrics


PADDLE_HITS_THRESHOLD = 100  # Number of paddles hits to consider the environment solved


LENGTH_PPO_CHOICE_HISTORY = 100
DUMMY_ACTION = -1
