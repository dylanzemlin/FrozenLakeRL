import gymnasium as gym
import numpy as np
import time

# Set the environment parameters
MAP_NAME = "8x8"
IS_SLIPPERY = True
RENDER_MODE = "none"

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
LEARNING_RATE = 0.8
DISCOUNT_FACTOR = 0.95
EPISODES = 100000
EXPLORATION_RATE = 0.25
EXPLORATION_DECAY = 0.0001

# outcomes = []
# time_per_episode = []
# last_episode_start = 0

def format_as_minutes_and_seconds(seconds):
    minutes = int(seconds / 60)
    seconds = int(seconds % 60)
    return f"{minutes}m {seconds}s"

def save_q_state():
    np.save("q_table.npy", Q)

def load_q_state():
    return np.load("q_table.npy")

# Start training
for i in range(EPISODES):
    # Print the current episode
    if i % 10000 == 0:
        print(f"\rEpisode {i+1}/{EPISODES}", end="")
    # if len(time_per_episode) > 0:
        # print(f" - ETA: {format_as_minutes_and_seconds(np.mean(time_per_episode) * (EPISODES - i))}")

    # last_episode_start = time.time()
    observation, rInfo = env.reset()
    # outcomes.append("Fail")
    while True:
        # While we are training, we want to explore the environment so we introduce some randomness into the actions
        # even if the Q table specifies a "better" action
        rng = np.random.rand()
        if rng < EXPLORATION_RATE or np.sum(Q[observation]) == 0:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[observation])

        # Take the action and observe the outcome
        nstate, reward, terminated, truncated, info = env.step(action)

        # Calculate the new Q value based on the observed outcome
        Q[observation, action] = Q[observation, action] + LEARNING_RATE * (reward + DISCOUNT_FACTOR * np.max(Q[nstate]) - Q[observation, action])

        # Update the observation
        observation = nstate

        # If the episode is terminated, we break the loop
        if terminated:
            # outcomes[-1] = "Success" if reward > 0 else "Fail"
            break

        if truncated:
            break

    EXPLORATION_RATE = max(EXPLORATION_RATE - EXPLORATION_DECAY, 0.1)

    # time_per_episode.append(time.time() - last_episode_start)

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

save_q_state()