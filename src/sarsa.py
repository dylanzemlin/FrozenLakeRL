import gymnasium as gym
import numpy as np

# Set the environment parameters
MAP_NAME = "8x8"
IS_SLIPPERY = True
RENDER_MODE = "none"

# Create the environment
env = gym.make("FrozenLake-v1", desc=None, map_name=MAP_NAME, is_slippery=IS_SLIPPERY, render_mode=RENDER_MODE)

# Reset the environment
env.reset()

# Setup the Q table and hyperparameters
Q = np.zeros([env.observation_space.n, env.action_space.n])
alpha = 0.05  
gamma = 0.99  
EPISODES = 100000
epsilon = 1.0
EXPLORATION_DECAY = 0.995  
MIN_EXPLORATION = 0.01 

def modify_reward(reward, terminated, info):
    if terminated and reward == 1:
        # Reward for reaching the goal
        return 50 
    elif terminated:
        # Penalty for falling into a hole
        return -1  
    else:
        # Small penalty for each step
        return 0

def save_q_state(Q):
    # Save Q-table to file
    np.save("q_table.npy", Q)

def load_q_state():
    # Load Q-table from file
    return np.load("q_table.npy")

total_reward = 0

# Start training with SARSA
for i in range(EPISODES):
    
    state = env.reset()[0]
    done = False
    
    while not done:
        # Choose action using epsilon-greedy policy
        if np.random.rand() < epsilon or np.sum(Q[state]) == 0:
            # Choose random action
            action = env.action_space.sample()
        else:
            # Choose the action with the highest value in the current state
            action = np.argmax(Q[state])
            
        # Take action and observe the outcome
        next_state, reward, terminated, truncated, info = env.step(action)
        
        # Custom reward modification
        modified_reward = modify_reward(reward, terminated, info)
        total_reward += modified_reward

        # Choose next action using epsilon-greedy policy
        if np.random.rand() < epsilon or np.sum(Q[next_state]) == 0:
            # Choose a random action
            next_action = env.action_space.sample()
        else:
            # Choose the action with the highest value in the next state
            next_action = np.argmax(Q[next_state])

        # Calculate Q(s, a) = Q(s, a) + alpha * (reward + gamma * Q(s', a') - Q(s, a))
        Q[state, action] = Q[state, action] + alpha * (modified_reward + gamma * Q[next_state, next_action] - Q[state, action])

        state = next_state
        action = next_action
        done = terminated or truncated

    # Adjust exploration rate using multicative decay
    epsilon = max(epsilon * EXPLORATION_DECAY, MIN_EXPLORATION)
    
    # Calculate and print the average reward
    avg_reward = total_reward / EPISODES
    print(f"\rEpisode {i+1}/{EPISODES}, Average Reward: {avg_reward:.2f}", end="")

episodes = 1000
nb_success = 0

# Evaluation
for _ in range(episodes):
    state = env.reset()[0]
    done = False
    
    while not done:
        # Choose the action with the highest value in the current state
        action = np.argmax(Q[state])
        
        # Implement this action and move the agent in the desired direction
        state, reward, terminated, truncated, info = env.step(action)
        
        done = terminated or truncated
        
        # Check if the episode was successful 
        if terminated and reward == 1: 
            nb_success += 1

# Print the success rate
print(f"\nSuccess rate = {nb_success / episodes * 100}%")

save_q_state(Q)








