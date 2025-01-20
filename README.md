# 2048-ppo-agent
![Testing](https://github.com/michaelriedl/2048-ppo-agent/actions/workflows/pytest.yml/badge.svg)
![Formatting](https://github.com/michaelriedl/2048-ppo-agent/actions/workflows/black.yml/badge.svg)
A proximal policy optimization (PPO) reinforcement learning (RL) agent for the game 2048.

## Introduction
This project is a reinforcement learning agent that uses the proximal policy optimization (PPO) algorithm to learn how to play the game 2048. The agent is implemented in Python using the PyTorch library. The gameplay environment leverages the Pgx library which is built with JAX
allowing for fast computation on GPUs.

## Getting Started
To get started, clone the repository and install the required dependencies using the following commands:
```bash
git clone https://github.com/michaelriedl/2048-ppo-agent.git
cd 2048-ppo-agent
conda create --name ppo-torch-env python=3.11
conda activate ppo-torch-env
pip install -r requirements.txt
```

## Naive Strategies
Before training the PPO agent, I implemented a few naive strategies to serve as baselines for comparison. 
These strategies include:
1. **Random Agent**: The random agent makes a random move at each step.
2. **Corner (DRUL) Agent**: The corner agent always prioritizes the moves in the order: Down, Right, Up, Left.

These srategies are explored in the notebook `notebooks/explore_naive_strategies.ipynb`. Some
visualizations of the strategies are shown below.

### Random Agent Visualization
<div align="center">
<figure>
<img src="./assets/2048_random_actions.svg" alt="Random Agent" style="width:50%">
</figure>
</div>

### Corner Agent Visualization
<div align="center">
<figure>
<img src="./assets/2048_drul_actions.svg" alt="Corner Agent" style="width:50%">
</figure>
</div>