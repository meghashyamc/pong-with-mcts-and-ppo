"""
Constants related to training
"""

from src.pong import constants

ARG_REWARD_FREQUENCY = "reward_frequency"
ARG_RENDER = "render"
ARG_MCTS_HEURISTIC = "heuristic"
ARG_PLOT = "plot"

USE_MCTS = "mcts"

TRAINED_FOLDER_NAME = "trained"
CSV_COLUMN_EPISODE = "Episode"
CSV_COLUMN_TIME = "Time (seconds)"
CSV_COLUMN_REWARD = "Average Reward"
CSV_COLUMN_PADDLE_HITS = "Average Paddle Hits"

STATE_DIMS = {
    constants.ENV_SIMPLE_PONG: 5,
    constants.ENV_COMPLEX_PONG: 7,
}

ACTION_DIMS = {
    constants.ENV_SIMPLE_PONG: 3,
    constants.ENV_COMPLEX_PONG: 9,
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
WINDOW_SIZE_FOR_MCTS = (
    300  # Number of episodes to average over for considering or not considering MCTS
)
PADDLE_HITS_THRESHOLD = 100  # Number of paddles hits to consider the environment solved

PPO_PERCENTAGE_THRESHOLD = 0.95

LENGTH_PPO_CHOICE_HISTORY = 100
FIELD_PPO_CHOICE_HISTORY = "ppo_choice_history"
