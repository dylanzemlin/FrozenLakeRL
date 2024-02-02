import gymnasium as gym

# Set the environment parameters
MAP_NAME = "8x8"
IS_SLIPPERY = False
RENDER_MODE = "human"

# Create the environment
env = gym.make("FrozenLake-v1", desc=None, map_name = MAP_NAME, is_slippery = IS_SLIPPERY, render_mode = RENDER_MODE)

# Reset the environment
env.reset()

# Render the environment
env.render()