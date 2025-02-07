# pylint: disable=no-member
"""
Functionality related to various game objects.
These objects can be used in other games - here they are specifically
using by Simple/Complex Pong games.
"""

import random
from typing import Tuple, List
from abc import ABC, abstractmethod
import pygame
from src.pong import constants
from src.models.pong import Velocity, Direction


class GameObject(ABC):
    """
    Abstract class with methods to be implemented by various game objects.
    """

    def __init__(self, x: float, y: float, width: float, height: float):
        self.width = width
        self.height = height
        self.position = pygame.Rect(x, y, width, height)
        self.velocity = Velocity(x=0, y=0)

    @abstractmethod
    def update(self, screen_rect: pygame.Rect):
        """
        Update the game object's state
        """

    @abstractmethod
    def copy(self) -> "GameObject":
        """
        Create a copy of the current game object
        """


class Paddle(GameObject):
    """
    Represents a paddle in the game that can move horizontally from left to right
    """

    @staticmethod
    def new(
        x: int = constants.SCREEN_WIDTH // 2 - constants.LARGE_PADDLE_WIDTH // 2,
        y: int = constants.SCREEN_HEIGHT - constants.PADDLE_HEIGHT - 10,
        width=constants.LARGE_PADDLE_WIDTH,
        height=constants.PADDLE_HEIGHT,
    ) -> "Paddle":
        """
        Create a new paddle at the center of the screen
        """
        return Paddle(
            x,
            y,
            width,
            height,
        )

    def __init__(
        self,
        x: float,
        y: float,
        width=constants.LARGE_PADDLE_WIDTH,
        height=constants.PADDLE_HEIGHT,
    ):
        super().__init__(x, y, width, height)

    def reset(self):
        """
        Resets the initial position of the paddle at the start of the game.
        """
        self.position.x = constants.SCREEN_WIDTH // 2 - self.width // 2
        self.position.y = constants.SCREEN_HEIGHT - self.height - 10

    def move(self, direction: Direction):
        """
        Move essentally changes the velocity of the paddle so that it can
        move to another position in the next update
        """
        self.velocity.x = direction.value * constants.PADDLE_SPEED

    def update(self, screen_rect: pygame.Rect, other_paddle: "Paddle" = None):
        """
        Updates the position of the paddle
        """
        new_x = self.position.x + self.velocity.x

        # Check collision with other paddle
        if other_paddle:
            if self.position.centerx < other_paddle.position.centerx:
                # This paddle is on the left
                new_x = min(new_x, other_paddle.position.left - self.width)
            else:
                # This paddle is on the right
                new_x = max(new_x, other_paddle.position.right)

        self.position.x = new_x
        self.position.clamp_ip(screen_rect)

    def copy(self) -> "Paddle":
        new_paddle = Paddle(self.position.x, self.position.y, self.width, self.height)
        new_paddle.velocity = Velocity(x=self.velocity.x, y=self.velocity.y)
        return new_paddle


class Obstacle(GameObject):
    """
    Represents an obstacle in the game that can move horizontally from left to right
    """

    @staticmethod
    def new() -> "Obstacle":
        """
        Create a new obstacle at the center of the screen
        """
        return Obstacle(
            constants.SCREEN_WIDTH // 2 - constants.SMALL_PADDLE_WIDTH // 2,
            constants.SCREEN_HEIGHT // 2 - constants.PADDLE_HEIGHT // 2,
        )

    def __init__(self, x: float, y: float):
        super().__init__(
            x, y, 2 * constants.LARGE_PADDLE_WIDTH, constants.PADDLE_HEIGHT
        )
        self.velocity.x = constants.OBSTACLE_SPEED

    def reset(self):
        """
        Resets the initial position of the obstacle at the start of the game.
        """
        self.position.x = constants.SCREEN_WIDTH // 2 - self.width // 2
        self.position.y = constants.SCREEN_HEIGHT // 2 - self.height // 2

    def move(self):
        """
        Sets the obstacle velocity
        """
        if self.position.left <= 0 or self.position.right >= constants.SCREEN_WIDTH:
            self.velocity.x *= -1

    def update(self, screen_rect: pygame.Rect):
        self.position.x += self.velocity.x

    def copy(self) -> "Obstacle":
        return Obstacle(self.position.x, self.position.y)


class Ball(GameObject):
    """
    Represents a ball in the game that can move in any direction.
    While the ball is actually a circle, we use a rectangle to represent it
    in terms of collision detection as that's how the ball essentially behaves
    in a rectangular screen.
    """

    @staticmethod
    def new() -> "Ball":
        """
        Create a new ball at the center of the screen
        """
        return Ball(
            constants.SCREEN_WIDTH // 2 - constants.BALL_SIZE // 2,
            constants.SCREEN_HEIGHT // 2 - constants.BALL_SIZE // 2,
        )

    def __init__(self, x: float, y: float, reset: bool = True):
        super().__init__(x, y, constants.BALL_SIZE, constants.BALL_SIZE)
        if reset:
            self.reset()

    def reset(self):
        """
        Resets the initial position of the ball at the start of the game.
        """
        self.position.x = random.randint(
            0, constants.SCREEN_WIDTH - constants.BALL_SIZE
        )
        self.position.y = 0

        self.velocity.x = random.choice(constants.BALL_SPEEDS_X) * random.choice(
            (1, -1)
        )
        self.velocity.y = random.choice(constants.BALL_SPEEDS_Y)

    def update(self, screen_rect: pygame.Rect):
        """
        Updates the position of the ball based on its velocity
        """
        self.position.x += self.velocity.x
        self.position.y += self.velocity.y

    def move(
        self, paddles: List[Paddle], obstacle: Obstacle = None
    ) -> Tuple[bool, int]:
        """
        Sets the ball velocity and returns the reward if the ball hits the paddle
        """
        if self.has_ball_hit_vertical_walls():
            self.velocity.x *= -1
            return False, 0

        if self.has_ball_hit_top_wall() and self.velocity.y < 0:
            # This ensures that once in a while, the ball's position is reset after
            # hitting the top wall. This is to ensure that the agent doesn't game the system
            # due to elastic collisions
            if random.random() < constants.BALL_RESET_PROBABILITY_WHEN_HITTING_TOP_WALL:
                self.reset()
            else:
                self.velocity.y *= -1
            return False, 0

        if self.has_ball_hit_paddle(paddles):
            self.velocity.y *= -1
            return False, constants.PADDLE_HIT_REWARD

        if self.has_ball_fallen_through():
            self.reset()
            return True, constants.BALL_FALLING_THROUGH_REWARD

        if obstacle and self.has_ball_hit_obstacle(obstacle):
            collision_side = self.get_collision_side(obstacle)
            if collision_side in (
                constants.OBSTACLE_SIDE_LEFT,
                constants.OBSTACLE_SIDE_RIGHT,
            ):
                # Reverse x velocity and add obstacle's velocity
                # because the ball's mass is assumed to be way less than
                # the obstacle's mass
                self.velocity.x = -self.velocity.x + 2 * obstacle.velocity.x
            elif collision_side in (
                constants.OBSTACLE_SIDE_TOP,
                constants.OBSTACLE_SIDE_BOTTOM,
            ):
                # Only reverse y velocity of the ball as the obstacle doesn't have any y velocity
                self.velocity.y = -self.velocity.y

            return False, 0

        return False, 0

    def get_collision_side(self, obstacle: Obstacle) -> str:
        """
        Determine which side of the obstacle the ball collided with
        """
        # Calculate the overlap on each side
        left_overlap = self.position.right - obstacle.position.left
        right_overlap = obstacle.position.right - self.position.left
        top_overlap = self.position.bottom - obstacle.position.top
        bottom_overlap = obstacle.position.bottom - self.position.top

        # Find the side with the smallest overlap
        min_overlap = min(left_overlap, right_overlap, top_overlap, bottom_overlap)

        if min_overlap == left_overlap:
            return constants.OBSTACLE_SIDE_LEFT
        if min_overlap == right_overlap:
            return constants.OBSTACLE_SIDE_RIGHT
        if min_overlap == top_overlap:
            return constants.OBSTACLE_SIDE_TOP
        return constants.OBSTACLE_SIDE_BOTTOM

    def has_ball_hit_vertical_walls(self) -> bool:
        """
        Check if the ball has hit the vertical walls
        """
        return (self.position.left <= 0 and self.velocity.x < 0) or (
            self.position.right >= constants.SCREEN_WIDTH and self.velocity.x > 0
        )

    def has_ball_hit_top_wall(self) -> bool:
        """
        Check if the ball has hit the top wall
        """
        return self.position.top <= 0

    def has_ball_hit_paddle(self, paddles: List[Paddle]) -> bool:
        """
        Check if the ball has hit the paddle
        """
        for paddle in paddles:
            if (
                (self.position.right >= paddle.position.left)
                and (self.position.left <= paddle.position.right)
                and (self.position.bottom >= paddle.position.y)
                and self.velocity.y > 0
            ):
                return True
        return False

    def has_ball_hit_obstacle(self, obstacle: Obstacle) -> bool:
        """
        Check if the ball has hit the obstacle
        """
        if (
            (self.position.right >= obstacle.position.left)
            and (self.position.left <= obstacle.position.right)
            and (self.position.bottom >= obstacle.position.top)
            and (self.position.top <= obstacle.position.bottom)
        ):
            return True

    def has_ball_fallen_through(self) -> bool:
        """
        Check if the ball has fallen through the bottom wall
        """
        return self.position.bottom >= constants.SCREEN_HEIGHT and self.velocity.y > 0

    def copy(self) -> "Ball":
        new_ball = Ball(self.position.x, self.position.y, reset=False)
        new_ball.velocity = Velocity(x=self.velocity.x, y=self.velocity.y)
        return new_ball
