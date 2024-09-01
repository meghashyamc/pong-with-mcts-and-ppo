# pylint: disable=no-member
"""
Functionality related to various game objects
"""

import random
import math
from typing import Tuple
from abc import ABC, abstractmethod
import pygame
from src.pong import constants
from src.models.pong import Velocity, Direction


class GameObject(ABC):
    """
    Abstract class with methods to be implemented by various game objects.
    """

    def __init__(self, x: float, y: float, width: float, height: float):
        self.position = pygame.Rect(x, y, width, height)
        self.velocity = Velocity(x=0, y=0)

    @abstractmethod
    def update(self, screen_rect: pygame.Rect):
        """
        Update the game object's state
        """


class Paddle(GameObject):
    """
    Represents a paddle in the game that can move horizontally from left to right
    """

    @staticmethod
    def new() -> "Paddle":
        """
        Create a new paddle at the center of the screen
        """
        return Paddle(
            constants.SCREEN_WIDTH // 2 - constants.PADDLE_WIDTH // 2,
            constants.SCREEN_HEIGHT - constants.PADDLE_HEIGHT - 10,
        )

    def __init__(self, x: float, y: float):
        super().__init__(x, y, constants.PADDLE_WIDTH, constants.PADDLE_HEIGHT)

    def reset(self):
        """
        Resets the initial position of the paddle at the start of the game.
        """
        self.position.x = constants.SCREEN_WIDTH // 2 - constants.PADDLE_WIDTH // 2
        self.position.y = constants.SCREEN_HEIGHT - constants.PADDLE_HEIGHT - 10

    def move(self, direction: Direction):
        """
        Move essentally changes the velocity of the paddle so that it can
        move to another position in the next update
        """
        self.velocity.x = direction.value * constants.PADDLE_SPEED

    def update(self, screen_rect: pygame.Rect):
        """
        Updates the position of the paddle
        """
        self.position.x += self.velocity.x
        self.position.clamp_ip(screen_rect)


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

    def __init__(self, x: float, y: float):
        super().__init__(x, y, constants.BALL_SIZE, constants.BALL_SIZE)
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

    def move(self, paddle: Paddle) -> Tuple[bool, int]:
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

        if self.has_ball_hit_paddle(paddle):
            self.velocity.y *= -1
            return False, constants.PADDLE_HIT_REWARD

        if self.has_ball_fallen_through():
            self.reset()
            return True, constants.BALL_FALLING_THROUGH_REWARD
        return False, 0

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

    def has_ball_hit_paddle(self, paddle: Paddle) -> bool:
        """
        Check if the ball has hit the paddle
        """

        return (
            (self.position.right >= paddle.position.left)
            and (self.position.left <= paddle.position.right)
            and (self.position.bottom >= paddle.position.y)
            and self.velocity.y > 0
        )

    def has_ball_fallen_through(self) -> bool:
        """
        Check if the ball has fallen through the bottom wall
        """
        return self.position.bottom >= constants.SCREEN_HEIGHT and self.velocity.y > 0


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
            constants.SCREEN_WIDTH // 2 - constants.PADDLE_WIDTH // 2,
            constants.SCREEN_HEIGHT // 2 - constants.PADDLE_HEIGHT // 2,
        )

    def __init__(self, x: float, y: float):
        super().__init__(x, y, constants.PADDLE_WIDTH, constants.PADDLE_HEIGHT)
        self.velocity.x = constants.OBSTACLE_SPEED

    def reset(self):
        """
        Resets the initial position of the obstacle at the start of the game.
        """
        self.position.x = constants.SCREEN_WIDTH // 2 - constants.PADDLE_WIDTH // 2
        self.position.y = constants.SCREEN_HEIGHT // 2 - constants.PADDLE_HEIGHT // 2

    def move(self):
        """
        Sets the obstacle velocity
        """
        if self.position.left <= 0 or self.position.right >= constants.SCREEN_WIDTH:
            self.velocity.x *= -1

    def update(self, screen_rect: pygame.Rect):
        self.position.x += self.velocity.x
