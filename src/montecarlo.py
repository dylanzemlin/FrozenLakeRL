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
rList = []
LEARNING_RATE = 1
EPISODES = 25000
EXPLORATION_RATE = 1
EXPLORATION_DECAY = 0.001
MAX_EXPLORATION_RATE = 1
MIN_EXPLORATION_RATE = 0.0001

# Start training
for i in range(EPISODES):
    # Print the progress
    if i % 10000 == 0 and i != 0:
        print(str(i) + "/" +str(EPISODES))

    # Reset the environment
    observation, rInfo = env.reset()

    # Initialize the variables
    x = 0
    y = []

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
        x += reward

        # Store the state and action for updating the Q table
        y.append((observation, action))

        # Update the observation
        observation = new_state

        # If the episode is terminated, we break the loop
        if terminated or truncated:
            break

    # Update the reward list
    rList.append(x)

    # Update the Q table using the state and action pairs as well as the reward for the episode
    for (state, action) in y:
        Q2[state, action] += 1
        LEARNING_RATE = 1 / Q2[state, action]
        Q[state, action] = Q[state, action] + LEARNING_RATE * (x - Q[state, action])

    # Update the exploration rate via a exponential decay function
    EXPLORATION_RATE = MIN_EXPLORATION_RATE + (MAX_EXPLORATION_RATE - MIN_EXPLORATION_RATE) * np.exp(-EXPLORATION_DECAY * i)

# Let's check our success rate!
print (f"Success rate = {sum(rList)/EPISODES}")