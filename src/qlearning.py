# Dylan Zemlin | 113562763
# Machine Learning (CS 5033)
# Reinforcement Learning Project | Q-Learning Algorithm

import gymnasium as gym
import numpy as np
import time

# Set the environment parameters
MAP_NAME = "8x8"
IS_SLIPPERY = True

# Create the environment
env = gym.make("FrozenLake-v1", desc=None, map_name = MAP_NAME, is_slippery = IS_SLIPPERY)

# Reset the environment
env.reset()

# Setup the Q table and hyperparameters
Q = np.zeros([env.observation_space.n, env.action_space.n])
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPISODES = 25000
EXPLORATION_RATE = 1
EXPLORATION_DECAY = 0.005
MAX_EXPLORATION_RATE = 1
MIN_EXPLORATION_RATE = 0.01


def format_as_minutes_and_seconds(seconds):
    minutes = int(seconds / 60)
    seconds = int(seconds % 60)
    return f"{minutes}m {seconds}s"


# Start the timer and initialize the exploration rates list
start_time = time.time()
exploration_rates = []

# Start training
for i in range(EPISODES):
    # Print the current episode
    if i % 10000 == 0:
        print(f"\rEpisode {i+1}/{EPISODES}", end="")

    observation, rInfo = env.reset()
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
            break

        if truncated:
            break

    # Exponential decay of the exploration rate
    EXPLORATION_RATE = MIN_EXPLORATION_RATE + (MAX_EXPLORATION_RATE - MIN_EXPLORATION_RATE) * np.exp(-EXPLORATION_DECAY * i)
    exploration_rates.append(EXPLORATION_RATE)
        
    # Linear decay of the exploration rate
    # EXPLORATION_RATE = EXPLORATION_RATE - EXPLORATION_DECAY
    # EXPLORATION_RATE = max(EXPLORATION_RATE, MIN_EXPLORATION_RATE)
    # EXPLORATION_RATE = min(EXPLORATION_RATE, MAX_EXPLORATION_RATE)
    # exploration_rates.append(EXPLORATION_RATE)

episodes = 1000
nb_success = 0

# Evaluation
for _ in range(episodes):
    observation, rInfo = env.reset()
    
    # Until the agent gets stuck or reaches the goal, keep running it
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

# Print the time it took to train the agent
end_time = time.time()
print(f"\nTraining took {format_as_minutes_and_seconds(end_time - start_time)}")

# Print the success rate of the agent
print (f"Success rate = {nb_success/episodes*100}%")