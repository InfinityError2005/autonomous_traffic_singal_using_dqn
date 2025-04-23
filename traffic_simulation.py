import random
import numpy as np

class TrafficSimulation:
    def __init__(self):
        # Initialize the traffic simulation with a state [north, south, east, west]
        self.state = [0, 0, 0, 0]  # [north_traffic, south_traffic, east_traffic, west_traffic]
        self.done = False
        self.steps = 0
        self.max_steps = 200  # Max steps per episode

        # Traffic light directions
        self.lights = ["North", "South", "East", "West"]

    def reset(self):
        # Reset the environment to a random initial state
        self.state = [random.randint(0, 10) for _ in range(4)]  # Random traffic flow in 4 directions
        self.done = False
        self.steps = 0
        return self.state

    def step(self, action):
        # Perform the action: change the light to the action index (0-3)
        self.change_traffic_light(action)

        # Calculate the reward based on the state (e.g., reduce traffic flow)
        next_state = self.get_state()
        reward = self.calculate_reward(next_state)
        self.done = self.check_done(next_state)

        return next_state, reward, self.done

    def change_traffic_light(self, direction):
        """
        Change the traffic light based on the action.
        The action is an integer representing the direction:
        0 = North, 1 = South, 2 = East, 3 = West
        """
        print(f"Traffic light changed to {self.lights[direction]}")  # Print action (for debugging)
        
        # Logic to handle light timing can be implemented here

    def get_state(self):
        """
        Return the current state of the traffic simulation.
        This will include the traffic flow in each direction.
        """
        return [random.randint(0, 10) for _ in range(4)]  # Example state: [north_traffic, south_traffic, east_traffic, west_traffic]

    def calculate_reward(self, state):
        """
        Reward function to encourage the agent to reduce congestion.
        The reward is negative if there is a lot of traffic, 
        and positive if traffic flow is smoother (lower congestion).
        """
        traffic_flow = sum(state[:4])  # Total traffic flow in all directions
        return -traffic_flow  # Negative reward for high traffic flow

    def check_done(self, state):
        """
        Check if the simulation episode should end.
        The episode ends when the maximum number of steps is reached.
        """
        self.steps += 1
        return self.steps >= self.max_steps

