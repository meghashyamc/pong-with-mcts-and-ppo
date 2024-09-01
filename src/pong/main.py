"""
Starting point of Simple/Complex Pong games when they are played by humans
"""

import argparse
from src.pong.base_game import BasePongGame
from src.pong import constants


def main():
    """
    Starting point of the Simple Pong game
    """

    parser = argparse.ArgumentParser(description="Play Pong game")

    # Add arguments
    parser.add_argument(
        f"--{constants.ARG_ENV_NAME}",
        type=str,
        default=constants.ENV_SIMPLE_PONG,
        help="Name of the environment",
    )

    args = parser.parse_args()
    game = BasePongGame.get_pong_game(env_name=args.env_name)
    game.run()


if __name__ == "__main__":
    main()
