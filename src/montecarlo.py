import gymnasium as gym
import numpy as np
import time

# Set the environment parameters
MAP_NAME = "4x4"
IS_SLIPPERY = True

# Create the environment
env = gym.make("FrozenLake-v1", desc=None, map_name = MAP_NAME, is_slippery = IS_SLIPPERY, max_episode_steps = 100)

# Reset the environment
env.reset()

# Setup the Q tables for the environment and hyperparameters
Q = np.zeros([env.observation_space.n, env.action_space.n])
Q2 = np.zeros([env.observation_space.n, env.action_space.n])
rList = []
LEARNING_RATE = 1
EPISODES = 500000
EXPLORATION_RATE = 1
EXPLORATION_DECAY = 0.001
MAX_EXPLORATION_RATE = 1
MIN_EXPLORATION_RATE = 0.0001

# Start training
for i in range(EPISODES):
    observation, rInfo = env.reset()
    x = 0
    y = []
    score = 0

    while True:
        rng = np.random.rand()
        if rng > EXPLORATION_RATE:
            action = np.argmax(Q[observation])
        else:
            action = env.action_space.sample()

        new_state, reward, terminated, truncated, info = env.step(action)
        x += reward
        score += reward
        y.append((observation, action))
        observation = new_state

        # If the episode is terminated, we break the loop
        if terminated:
            break

        if truncated:
            break

    rList.append(x)

    for (state, action) in y:
        Q2[state, action] += 1
        LEARNING_RATE = 1 / Q2[state, action]
        Q[state, action] = Q[state, action] + LEARNING_RATE * (score - Q[state, action])

    if i % 10000 == 0 and i != 0:
        print(str(i) + "/" +str(EPISODES))
        print("Current score: " + str(sum(rList) / i))

    EXPLORATION_RATE = MIN_EXPLORATION_RATE + (MAX_EXPLORATION_RATE - MIN_EXPLORATION_RATE) * np.exp(-EXPLORATION_DECAY * i)

# Let's check our success rate!
print (f"Success rate = {sum(rList)/EPISODES}")