import gymnasium as gym
import numpy as np
import time

# Set the environment parameters
MAP_NAME = "8x8"
IS_SLIPPERY = False
RENDER_MODE = "human"

# Helper ENUM for action space
class Action:
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

# Create the environment
env = gym.make("FrozenLake-v1", desc=None, map_name = MAP_NAME, is_slippery = IS_SLIPPERY, render_mode = RENDER_MODE)

# Reset the environment
env.reset()

# Setup the Q table and hyperparameters
Q = np.zeros([env.observation_space.n, env.action_space.n])

def load_q_state():
    return np.load("q_table.npy")

Q = load_q_state()

episodes = 100
nb_success = 0

# Evaluation
for _ in range(episodes):
    observation, rInfo = env.reset()
    
    # Until the agent gets stuck or reaches the goal, keep training it
    while True:
        # Choose the action with the highest value in the current state
        action = np.argmax(Q[observation])

        # Implement this action and move the agent in the desired direction
        new_state, reward, terminated, truncated, info = env.step(action)

        # Update our current state
        observation = new_state

        # When we get a reward, it means we solved the game
        nb_success += reward

        if terminated:
            break

        if truncated:
            break

# Let's check our success rate!
print (f"Success rate = {nb_success/episodes*100}%")