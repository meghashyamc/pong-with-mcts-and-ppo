import torch
import pygame
from src.pong.simple.game import PongGame
from src.algorithms.PPO import PPO
from src.logger.logger import logger


def visualize_trained_ppo(model_path, num_episodes=5):
    # Initialize the game
    game = PongGame(headless=False)

    # Set up the PPO agent
    state_dim = 5
    action_dim = 3

    ppo_agent = PPO(
        state_dim,
        action_dim,
        lr_actor=0.0003,
        lr_critic=0.001,
        gamma=0.99,
        K_epochs=80,
        eps_clip=0.2,
        has_continuous_action_space=False,
    )

    # Load the trained model
    ppo_agent.load(model_path)

    for episode in range(num_episodes):
        state = game.reset()
        done = False
        episode_reward = 0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # Select action
            action = ppo_agent.select_action(state)

            # Perform action in the environment
            state, reward, done, _ = game.step(action)

            episode_reward += reward

            # Render the game
            game.render()
            pygame.time.wait(50)  # Add a small delay to slow down the visualization

        logger.info("Episode %d Reward: %.6f", episode + 1, episode_reward)

    pygame.quit()


if __name__ == "__main__":
    model_path = "trained/SimplePong/PPO_SimplePong_final.pth"
    visualize_trained_ppo(model_path)
