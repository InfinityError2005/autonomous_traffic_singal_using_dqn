Overview
-----------
This project uses Deep Q Network (DQN) to control traffic signals at a 4-way intersection. The system learns to optimize traffic light timings based on real-time traffic flow, reducing congestion and improving traffic efficiency.

Files
----------
traffic_simulation.py: Simulates the traffic intersection, vehicles, and traffic signals.
dqn_agent.py: Implements the DQN agent that learns the optimal signal timings.
main.py: Integrates the traffic simulation and DQN agent to run the simulation.

Setup
-------
git clone https://github.com/Praveen121-ux/Reinforcement-Learning-Traffic-Signal-Using-DQN.git
cd Autonomous-Traffic-Signal-DQN

Install Dependencies
----------------------
pip install -r requirements.txt

Dependencies include:
----------------------
pygame
torch
numpy

How It Works
---------------------
The DQN agent learns to control the traffic signals at a 4-way intersection based on vehicle count and traffic conditions.

The agent optimizes signal timings to minimize congestion and improve traffic flow.

The simulation uses pygame for visualization.

Run the Simulation
--------------------
python main.py

Attribution
---------------
This project is based on the **Traffic Intersection Simulation** by **Mihir M. Gandhi** (https://github.com/mihir-m-gandhi/Traffic-Intersection-Simulation-with-Turns) with modifications to implement a **DQN-based traffic signal controller**.

this is for education use

