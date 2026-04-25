# Bus Routing Reinforcement Learning

A research-oriented project implementing a custom Gymnasium environment and reinforcement learning agents for Duke University campus bus route management. The goal is to learn dispatch and holding policies that reduce passenger waiting time while balancing active bus deployment.

## What it Does

This repository simulates a multi-stop bus network and trains agents to manage buses over an 8-hour day. The environment uses a queue-based passenger model, time-varying arrival spikes, and four discrete actions: normal operation, add a bus, remove a bus, or hold a bus. A DQN agent is implemented in `src/agent.py`, and training scripts for both DQN and PPO are provided.

## Project Structure
Project Demo: Link
Project Technical Walkthrough: Link

## Project Structure

- `src/env.py` - Custom Gymnasium `BusEnv` environment that simulates bus movement, passenger arrivals, queue dynamics, and reward shaping.
- `src/agent.py` - Deep Q-Network agent with replay buffer, target network updates, epsilon-greedy exploration, and an observation encoder for the bus fleet state.
- `src/train_dqn.py` - Training pipeline for the DQN agent; saves a checkpoint to `model/dqn_agent.pth`.
- `src/train_ppo.py` - PPO training script using `stable-baselines3`; saves the trained model to `model/ppo_agent.pth`.
- `model/` - Saved model checkpoints and serialized agents.
- `notebooks/` - Experimentation and visualization notebooks for DQN/ PPO hyperparameters and single-episode data.
- `slurm/` - Slurm job submission scripts for DQN and PPO training.

## Quick Start

1. Create and activate a Python environment.

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies.

   ```bash
   pip install -r requirements.txt
   pip install stable-baselines3
   ```

3. Run DQN training.

   ```bash
   python src/train_dqn.py
   ```

4. Run PPO training.

   ```bash
   python src/train_ppo.py
   ```

5. Optionally inspect experiment results in `notebooks/single-episode-data-visualization.ipynb`.

6. Optionally run training frameworks with CUDA.

    ```bash
    sbatch slurm/train_dqn.sh
    ```

    ```bash
    sbatch slrum/train_ppo.sh
    ```

    Ouput logs will saved to `slurm-logs/`

## Running the Environment

The environment is registered inside both training scripts as `gymnasium_env/BusRouting-v0`. To create it manually, use:

```python
import gymnasium as gym
from src.env import BusEnv

gym.register(id="gymnasium_env/BusRouting-v0", entry_point=BusEnv, max_episode_steps=50000)
env = gym.make("gymnasium_env/BusRouting-v0")
``` 

## Evaluation

The repository does not include fixed benchmark results in the README, but see `EVALUATION.md` for qualitative and quantitative evaluation metrics.

## Individual Contributions

Regan
- Custom environment and agent coding
- Notebook hyperparameter experimentation
- Project documentation (README, ATTRIBUTION, SETUP)
- Slurm job scheduling

Davidovitch
- Qualitative report of hyperparameter and architectural study (EVALUATION)
- Environment (reward, actions, states) justification