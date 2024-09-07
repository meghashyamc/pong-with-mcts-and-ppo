"""
Starting point of Simple/Complex Pong games when they are played by humans
"""

import argparse
from src.pong.game_factory import get_pong_game
from src.pong import constants


def main():
    """
    Starting point of Pong game
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
    game = get_pong_game(env_name=args.env_name)()
    game.run()


if __name__ == "__main__":
    main()
