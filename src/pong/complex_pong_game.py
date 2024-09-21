# pylint: disable=no-member
"""
Functionality for combining the various parts of the Complex Pong game
"""
import copy
from typing import List, Tuple
import pygame
from src.pong import constants
from src.pong.game_object import Paddle, Ball, Obstacle
from src.models.pong import Direction
from src.pong.base_game import BasePongGame


class ComplexPongGame(BasePongGame):
    """
    Complex Pong game class
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
            pygame.display.set_caption(constants.SCREEN_CAPTION_COMPLEX_PONG)
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, constants.GAME_FONT_SIZE)
        else:
            # Create a mock surface for collision detection
            self.screen = pygame.Surface(
                (constants.SCREEN_WIDTH, constants.SCREEN_HEIGHT)
            )

        self.paddle1 = Paddle.new(x=0, width=constants.SMALL_PADDLE_WIDTH)
        self.paddle2 = Paddle.new(
            x=constants.SCREEN_WIDTH // 2, width=constants.SMALL_PADDLE_WIDTH
        )
        self.ball = Ball.new()
        self.obstacle = Obstacle.new()
        self.reward = 0
        self.done = False

    def reset(self) -> List[float]:
        """Reset the game state and return the initial state."""
        self.paddle1.reset()
        self.paddle2.reset()
        self.ball.reset()
        self.obstacle.reset()
        self.reward = 0
        self.done = False
        return self._get_state()

    def step(self, action: int) -> Tuple[List, float, bool]:
        """
        Take a step in the environment.

        Args:
            action (int): 0 for left, left, 1 for stay, left, 2 for right, left
            4 for left, stay, 5 for right, stay, 6 for right, stay
            7 for left, right, 8 for right, right, 9 for right, right

        Returns:
            tuple: (next_state, reward, done)
        """

        match (action):
            case constants.COMPLEX_PONG_ACTION_PADDLE1_LEFT_PADDLE2_LEFT:
                self.paddle1.move(Direction.LEFT)
                self.paddle2.move(Direction.LEFT)
            case constants.COMPLEX_PONG_ACTION_PADDLE1_STAY_PADDLE2_LEFT:
                self.paddle1.move(Direction.STAYPUT)
                self.paddle2.move(Direction.LEFT)
            case constants.COMPLEX_PONG_ACTION_PADDLE1_RIGHT_PADDLE2_LEFT:
                self.paddle1.move(Direction.RIGHT)
                self.paddle2.move(Direction.LEFT)
            case constants.COMPLEX_PONG_ACTION_PADDLE1_LEFT_PADDLE2_STAY:
                self.paddle1.move(Direction.LEFT)
                self.paddle2.move(Direction.STAYPUT)
            case constants.COMPLEX_PONG_ACTION_PADDLE1_STAY_PADDLE2_STAY:
                self.paddle1.move(Direction.STAYPUT)
                self.paddle2.move(Direction.STAYPUT)
            case constants.COMPLEX_PONG_ACTION_PADDLE1_RIGHT_PADDLE2_STAY:
                self.paddle1.move(Direction.RIGHT)
                self.paddle2.move(Direction.STAYPUT)
            case constants.COMPLEX_PONG_ACTION_PADDLE1_LEFT_PADDLE2_RIGHT:
                self.paddle1.move(Direction.LEFT)
                self.paddle2.move(Direction.RIGHT)
            case constants.COMPLEX_PONG_ACTION_PADDLE1_STAY_PADDLE2_RIGHT:
                self.paddle1.move(Direction.STAYPUT)
                self.paddle2.move(Direction.RIGHT)
            case constants.COMPLEX_PONG_ACTION_PADDLE1_RIGHT_PADDLE2_RIGHT:
                self.paddle1.move(Direction.RIGHT)
                self.paddle2.move(Direction.RIGHT)
            case _:
                raise ValueError(f"Invalid action: {action}")
        self.update()
        next_state = self._get_state()

        return next_state, self.reward, self.done

    def update(self):
        """Update game state."""
        self.paddle1.update(self.screen.get_rect(), self.paddle2)
        self.paddle2.update(self.screen.get_rect(), self.paddle1)
        self.ball.update(self.screen.get_rect())
        self.obstacle.update(self.screen.get_rect())
        self.obstacle.move()
        done, reward = self.ball.move([self.paddle1, self.paddle2], self.obstacle)
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
            min(
                abs(self.ball.position.centerx - self.paddle1.position.centerx),
                abs(self.ball.position.centerx - self.paddle2.position.centerx),
            )
            / constants.SCREEN_WIDTH
        )

    def _get_state(self) -> List[float]:
        """
        Get the current state of the game.

        Returns:
            list: [paddle_x, ball_x, ball_y, ball_velocity_x, ball_velocity_y]
        """
        return [
            self.paddle1.position.centerx / constants.SCREEN_WIDTH,
            self.paddle2.position.centerx / constants.SCREEN_WIDTH,
            self.ball.position.centerx / constants.SCREEN_WIDTH,
            self.ball.position.centery / constants.SCREEN_HEIGHT,
            self.ball.velocity.x / constants.SCREEN_WIDTH,
            self.ball.velocity.y / constants.SCREEN_WIDTH,
            self.obstacle.position.centerx / constants.SCREEN_WIDTH,
        ]

    def render(self):
        """Render the current game state."""
        if self.headless:
            return
        self.screen.fill(constants.BLACK)
        pygame.draw.rect(self.screen, constants.WHITE, self.paddle1.position)
        pygame.draw.rect(self.screen, constants.WHITE, self.paddle2.position)
        pygame.draw.ellipse(self.screen, constants.WHITE, self.ball.position)
        pygame.draw.rect(self.screen, constants.WHITE, self.obstacle.position)

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
                        self.paddle1.move(Direction.LEFT)
                    elif event.key == pygame.K_RIGHT:
                        self.paddle1.move(Direction.RIGHT)
                    if event.key == pygame.K_a:
                        self.paddle2.move(Direction.LEFT)
                    elif event.key == pygame.K_d:
                        self.paddle2.move(Direction.RIGHT)

                if event.type == pygame.KEYUP:
                    if event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                        self.paddle1.move(Direction.STAYPUT)
                    if event.key in (pygame.K_a, pygame.K_d):
                        self.paddle2.move(Direction.STAYPUT)

            self.update()
            self.render()
            self.clock.tick(constants.FPS)

            if self.done:
                self.reset()

    def clone(self) -> "ComplexPongGame":
        """Create a deep copy of the current game instance."""
        new_game = ComplexPongGame(True, self.reward_frequency)
        new_game.paddle1 = self.paddle1.copy()
        new_game.paddle2 = self.paddle2.copy()
        new_game.obstacle = self.obstacle.copy()
        new_game.ball = self.ball.copy()
        new_game.reward = self.reward
        new_game.done = self.done
        return new_game

    def get_action_based_on_heuristic(self, randomness=0.1) -> int:
        """
        Gets the action to take based on the relative position of the
        ball and paddles.
        """
        if self.ball.position.centerx < self.paddle1.position.left:
            return constants.COMPLEX_PONG_ACTION_PADDLE1_LEFT_PADDLE2_STAY
        if (
            self.ball.position.centerx >= self.paddle1.position.left
            and self.ball.position.centerx <= self.paddle1.position.right
        ):
            return constants.COMPLEX_PONG_ACTION_PADDLE1_STAY_PADDLE2_STAY
        if (
            self.ball.position.centerx > self.paddle1.position.right
            and self.ball.position.centerx < self.paddle2.position.left
        ):
            return constants.COMPLEX_PONG_ACTION_PADDLE1_RIGHT_PADDLE2_LEFT
        if (
            self.ball.position.centerx >= self.paddle2.position.left
            and self.ball.position.centerx <= self.paddle2.position.right
        ):
            return constants.COMPLEX_PONG_ACTION_PADDLE1_STAY_PADDLE2_STAY

        if self.ball.position.centerx > self.paddle2.position.right:
            return constants.COMPLEX_PONG_ACTION_PADDLE1_STAY_PADDLE2_RIGHT

        return constants.COMPLEX_PONG_ACTION_PADDLE1_STAY_PADDLE2_STAY
