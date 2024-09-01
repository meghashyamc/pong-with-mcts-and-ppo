"""
Constants related to training
"""

from src.pong import constants

ARG_REWARD_FREQUENCY = "reward_frequency"


TRAINED_FOLDER_NAME = "trained"
CSV_COLUMN_EPISODE = "Episode"
CSV_COLUMN_TIME = "Time (seconds)"
CSV_COLUMN_REWARD = "Average Reward"
CSV_COLUMN_PADDLE_HITS = "Average Paddle Hits"

STATE_DIMS = {
    constants.ENV_SIMPLE_PONG: 5,
    constants.ENV_COMPLEX_PONG: 5,
}

ACTION_DIMS = {
    constants.ENV_SIMPLE_PONG: 3,
    constants.ENV_COMPLEX_PONG: 3,
}
