import gymnasium as gym
import numpy as np

# Set the environment parameters
MAP_NAME = "8x8"
IS_SLIPPERY = True
RENDER_MODE = "none"

# Create the environment 
env = gym.make("FrozenLake-v1", desc=None, map_name=MAP_NAME, is_slippery=IS_SLIPPERY, render_mode=RENDER_MODE)

# Setup value function, policy, and hyperparameters
V = np.zeros(env.observation_space.n)  
policy = np.zeros([env.observation_space.n, env.action_space.n])
EPISODES = 10000
gamma = 0.99  
theta = 0.0001  

def calculate_action_values(V, state):
    A = np.zeros(env.action_space.n)
    for action in range(env.action_space.n):
        for probability, next_state, reward, terminated in env.P[state][action]:
            # Update action value with Bellman equation
            A[action] += probability * (reward + gamma * V[next_state])
    return A

# Value iteration
while True:
    # Delta to keep track of maximum change in value
    delta = 0 
    for state in range(env.observation_space.n):
        # Calculate action value for each state
        A = calculate_action_values(V, state)
        # Get action value
        best_action_value = np.max(A)
        # Update delta
        delta = max(delta, best_action_value - V[state])
        # Update value function
        V[state] = best_action_value
    
    # If convergence is reached
    if delta < theta:
        break

# Optimal policy after convergence
for state in range(env.observation_space.n):
    # Calculate action value for each state
    A = calculate_action_values(V, state)
    # Get best action
    best_action = np.argmax(A)
    # Mark best action with 1
    policy[state, best_action] = 1


nb_success = 0
total_reward = 0

# Evaluation
for _ in range(EPISODES):
    state = env.reset()[0]  
    done = False
    
    while not done:
        # Choose the action based off policy  
        action = np.argmax(policy[state])  
        
        # Implement this action and move the agent in the desired direction
        state, reward, terminated, truncated, info = env.step(action)
        
        # Update reward for episode 
        total_reward += reward
        
        done = terminated or truncated  
        
        # Check if the episode was successful 
        if terminated and reward == 1:  
            nb_success += 1  
    

# Print the success rate and average reward
print(f"\nSuccess rate = {nb_success / EPISODES * 100}%")
print(f"Average reward = {total_reward / EPISODES}")