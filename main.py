import pygame
from dqn_agent import DQNAgent
from simulation import TrafficSimulation

# Initialize the Pygame window
pygame.init()

# Create an instance of the traffic simulation
env = TrafficSimulation()

# Define state size and action size
state_size = 4  # We have 4 traffic directions (North, South, East, West)
action_size = 4  # 4 possible actions: change light to North, South, East, or West

# Initialize the DQN agent
agent = DQNAgent(state_size=state_size, action_size=action_size)

# Training loop for the DQN agent
for episode in range(1000):  # Run 1000 episodes of training
    state = env.reset()  # Reset the environment to get the initial state
    done = False
    total_reward = 0

    # Start stepping through the environment
    while not done:
        action = agent.act(state)  # Get the action from the agent (which traffic light to change)
        next_state, reward, done = env.step(action)  # Execute the action in the simulation

        # Store the experience in memory and train the agent
        agent.remember(state, action, reward, next_state, done)
        agent.train()

        # Update the current state and accumulate total reward
        state = next_state
        total_reward += reward

    # Print the total reward for this episode
    print(f"Episode {episode + 1}/1000, Total Reward: {total_reward}")

    # Update the target model every few episodes
    if episode % 10 == 0:
        agent.update_target()

# Optionally, you can visualize the environment or save the model here

# Quit Pygame when finished
pygame.quit()
