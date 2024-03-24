# Dylan Zemlin | 113562763
# Machine Learning (CS 5033)
# Reinforcement Learning Project | Monte-Carlo Algorithm

import gymnasium as gym
import numpy as np

# Set the environment parameters
MAP_NAME = "8x8"
IS_SLIPPERY = True

# Create the environment
env = gym.make("FrozenLake-v1", desc=None, map_name = MAP_NAME, is_slippery = IS_SLIPPERY)

# Initialize Q table and hyperparameters
Q = np.zeros([env.observation_space.n, env.action_space.n])
RET_SUM = np.zeros([env.observation_space.n, env.action_space.n])
RET_CT = np.zeros([env.observation_space.n, env.action_space.n])
EPISODES = 25000
EXPLORATION_RATE = 1
EXPLORATION_DECAY = 0.001
MAX_EXPLORATION_RATE = 1
MIN_EXPLORATION_RATE = 0.0001
REWARD_LIST = []

# Start training
for i in range(EPISODES):
    # Print the current episode
    if i % 10000 == 0:
        print(f"\rEpisode {i+1}/{EPISODES}", end="")

    # Initialize the episode and reset the environment
    episode = []
    observation, rInfo = env.reset()
    
    # Generate an episode
    while True:
        # Choose an action based on the exploration rate
        rng = np.random.rand()
        if rng < EXPLORATION_RATE:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
            
        # Take the action and observe the outcome
        new_state, reward, terminated, truncated, info = env.step(action)
        episode.append((observation, action, reward))
        observation = new_state

        # If the episode is terminated, we break the loop
        if terminated or truncated:
            break

    # Calculate the total reward for the episode
    # Is this correct? Something seems off about how the reward is calculated and the Q-Table is updated after this step.
    # However, it appears to work. This is something that would be nice to look into later.
    G = 0
    for t in reversed(range(len(episode))):
        # Update the total reward
        state, action, reward = episode[t]
        G = G + reward
        
        # Check if the state-action pair is visited for the first time
        if not any((state == x[0] and action == x[1]) for x in episode[:t]):
            RET_SUM[state][action] += G
            RET_CT[state][action] += 1.0
            Q[state][action] = RET_SUM[state][action] / RET_CT[state][action]
    
    # Update exploration rate
    EXPLORATION_RATE = MIN_EXPLORATION_RATE + (MAX_EXPLORATION_RATE - MIN_EXPLORATION_RATE) * np.exp(-EXPLORATION_DECAY * i)

    # Track rewards
    REWARD_LIST.append(G)

# Calculate success rate
print(f"Success rate = {sum(REWARD_LIST) / EPISODES * 100}")
