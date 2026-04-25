# Attribution

This repository is built using the following open-source libraries and resources:

- `gymnasium` for the custom reinforcement learning environment API and environment lifecycle.
- `numpy` for numerical arrays, random passenger generation, and state representation.
- `torch` for neural network implementation, training, and optimization in the DQN agent.
- `stable-baselines3` for the PPO training pipeline and baseline reinforcement learning implementation.

## Code and Implementation

- The core environment implementation is in `src/env.py`, which simulates bus stops, queues, passenger arrivals, and reward shaping.
- The deep Q-network agent is implemented in `src/agent.py`, including replay buffer support, target network updates, epsilon-greedy exploration, and Huber loss.
- Training scripts are provided in `src/train_dqn.py` and `src/train_ppo.py`.

## Models and Data

- No external dataset is included in this repository. The environment generates synthetic passenger arrival data during simulation.
- Saved model checkpoints are stored under `model/`.

## AI Assistance

- AI agents via VSCode were used for the following tasks
    - dependency debugging throughout the codebase (several dependency errors were resolved by prompting VSCode Agent)
    - Gymnasium API syntax and class structure explanation (asked for documentation simplification and necessary structure to implement API)
    - Comment creation (asked VSCode to create comments for functions and varios in-line explanations)

- ChatGPT 5.2 was used for the following tasks
    - project brainstorming and research suggestions (prompting with backgorund and interests as well as project rubric)
    - custom environment creation guidelines and limitations to direct our environment development (training feasability, level of abstraction from real world scenarioes, most important values to consider in reward function)

Function specific input from AI agents is listed within each file of the codebase.

Code Attribution:

src/agent.py
- debugging gymnasium API by prompting VSCode Agent
- render() function was inspired by ChatGPT 5.2 and brainstorming potential outputs

src/env.py
- helper functions and code extensability was assisted by ChatGPT 5.2

src/train_dqn.py

src/train_ppo.py