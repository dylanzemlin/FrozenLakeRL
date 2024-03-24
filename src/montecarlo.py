# Dylan Zemlin | 113562763
# Machine Learning (CS 5033)
# Reinforcement Learning Project | Monte Carlo Algorithm

import gymnasium as gym
import numpy as np

# Set the environment parameters
MAP_NAME = "8x8"
IS_SLIPPERY = True

# Create the environment
env = gym.make("FrozenLake-v1", desc=None, map_name = MAP_NAME, is_slippery = IS_SLIPPERY)

# Reset the environment
env.reset()

# Setup the Q tables for the environment and hyperparameters
Q = np.zeros([env.observation_space.n, env.action_space.n])
Q2 = np.zeros([env.observation_space.n, env.action_space.n])
REWARD_LIST = []
LEARNING_RATE = 1
EPISODES = 25000
EXPLORATION_RATE = 1
EXPLORATION_DECAY = 0.001
MAX_EXPLORATION_RATE = 1
MIN_EXPLORATION_RATE = 0.0001

# Start training
for i in range(EPISODES):
    # Reset the environment
    observation, rInfo = env.reset()

    # Print the progress
    if i % 10000 == 0:
        print(f"\rEpisode {i+1}/{EPISODES}", end="")
    
    # Initialize the variables
    X = 0
    Y = []

    while True:
        # Choose an action by picking from the Q table or randomly
        rng = np.random.rand()
        if rng > EXPLORATION_RATE:
            action = np.argmax(Q[observation])
        else:
            action = env.action_space.sample()

        # Take the action and observe the outcome
        new_state, reward, terminated, truncated, info = env.step(action)

        # Update the reward for the episode
        X += reward

        # Store the state and action for updating the Q table
        Y.append((observation, action))

        # Update the observation
        observation = new_state

        # If the episode is terminated, we break the loop
        if terminated or truncated:
            break

    # Update the reward list
    REWARD_LIST.append(X)

    # Update the Q table using the state and action pairs as well as the reward for the episode
    # Should this happen every few episodes and not just per episode? This feels more "Q-Learny" rather than something Monte-Carlo based
    # Will need some more research in the future
    for (state, action) in Y:
        # Update the learning rate such that it adapts based on secondary q table (with a bit of magic number)
        LEARNING_RATE = 1 / Q2[state, action] * 0.975
        # Increment the secondary q table by a bit such that the learning rate can slowly adapt as needed based on the other observations
        Q2[state, action] += 0.835
        # Update the Q-Table using the current q value + learning rate times a reward sum with a bit of magic number
        Q[state, action] = Q[state, action] + LEARNING_RATE * (X - Q[state, action] * 0.985)

    # Update the exploration rate via a exponential decay function
    EXPLORATION_RATE = MIN_EXPLORATION_RATE + (MAX_EXPLORATION_RATE - MIN_EXPLORATION_RATE) * np.exp(-EXPLORATION_DECAY * i)

# Let's check our success rate!
print (f"Success rate = {sum(REWARD_LIST) / EPISODES}")
