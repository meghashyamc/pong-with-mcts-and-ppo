# pylint: disable=no-member
"""
Functionality for combining the various parts of the Simple Pong game
"""
from typing import List, Tuple
import pygame
from src.pong import constants
from src.pong.game_object import Paddle, Ball
from src.models.pong import Direction
from src.pong.base_game import BasePongGame


class SimplePongGame(BasePongGame):
    """
    Simple Pong game class
    """

    def __init__(
        self,
        headless: bool = False,
        reward_frequency: str = constants.FREQUENCY_FREQUENT,
    ):
        self.headless = headless
        self.reward_frequency = reward_frequency
        if not headless:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (constants.SCREEN_WIDTH, constants.SCREEN_HEIGHT)
            )
            pygame.display.set_caption(constants.SCREEN_CAPTION)
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, constants.GAME_FONT_SIZE)
        else:
            # Create a mock surface for collision detection
            self.screen = pygame.Surface(
                (constants.SCREEN_WIDTH, constants.SCREEN_HEIGHT)
            )

        self.paddle = Paddle.new()
        self.ball = Ball.new()
        self.reward = 0
        self.done = False

    def reset(self):
        """Reset the game state and return the initial state."""
        self.paddle.reset()
        self.ball.reset()
        self.reward = 0
        self.done = False
        return self._get_state()

    def step(self, action) -> Tuple[List, float, bool]:
        """
        Take a step in the environment.

        Args:
            action (int): 0 for left, 1 for stay, 2 for right

        Returns:
            tuple: (next_state, reward, done)
        """
        if action == 0:
            self.paddle.move(Direction.LEFT)
        elif action == 2:
            self.paddle.move(Direction.RIGHT)
        else:
            self.paddle.move(Direction.STAYPUT)

        self.update()
        next_state = self._get_state()

        return next_state, self.reward, self.done

    def update(self):
        """Update game state."""
        self.paddle.update(self.screen.get_rect())
        self.ball.update(self.screen.get_rect())
        done, reward = self.ball.move(self.paddle)
        self.update_reward(reward)
        self.done = done

    def update_reward(self, reward: float):
        """Update the reward based on the reward received or the default reward."""
        if not reward and self.reward_frequency == constants.FREQUENCY_FREQUENT:
            reward = self.get_default_reward()
        self.reward = reward

    def get_default_reward(self) -> float:
        """Get the default reward for an action."""

        return 1 - (
            abs(self.ball.position.centerx - self.paddle.position.centerx)
            / constants.SCREEN_WIDTH
        )

    def _get_state(self):
        """
        Get the current state of the game.

        Returns:
            list: [paddle_x, ball_x, ball_y, ball_velocity_x, ball_velocity_y]
        """
        return [
            self.paddle.position.centerx / constants.SCREEN_WIDTH,
            self.ball.position.centerx / constants.SCREEN_WIDTH,
            self.ball.position.centery / constants.SCREEN_HEIGHT,
            self.ball.velocity.x / constants.SCREEN_WIDTH,
            self.ball.velocity.y / constants.SCREEN_WIDTH,
        ]

    def render(self):
        """Render the current game state."""
        if self.headless:
            return
        self.screen.fill(constants.BLACK)
        pygame.draw.rect(self.screen, constants.WHITE, self.paddle.position)
        pygame.draw.ellipse(self.screen, constants.WHITE, self.ball.position)

        reward_surface = self.font.render(
            f"reward: {self.reward}", True, constants.WHITE
        )
        self.screen.blit(reward_surface, (10, 10))

        pygame.display.flip()

    def close(self):
        """Close the Pygame window."""
        pygame.quit()

    def run(self):
        """Main game loop for human play."""
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.paddle.move(Direction.LEFT)
                    elif event.key == pygame.K_RIGHT:
                        self.paddle.move(Direction.RIGHT)
                if event.type == pygame.KEYUP:
                    if event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                        self.paddle.move(Direction.STAYPUT)

            self.update()
            self.render()
            self.clock.tick(constants.FPS)

            if self.done:
                self.reset()
