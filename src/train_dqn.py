import os
import gymnasium as gym
import numpy as np
import torch
from env import BusEnv
from agent import BusAgent

def train(agent, env, n_episodes=1000, print_every=10, timestep_penalty=0.2):
    """
    Train a Q-learning agent.

    Args:
        agent: BusAgent instance to train
        env: Gymnasium environment (should be wrapped with RecordEpisodeStatistics)
        n_episodes: Number of episodes to train for
        print_every: Print progress every N episodes (set to None to disable printing)

    Returns:
        agent: The trained agent
        env: The environment with recorded statistics
    """
    for episode in range(n_episodes):
        state, info = env.reset()
        episode_over = False

        while not episode_over:
            action = agent.action_select(state)

            next_state, reward, terminated, truncated, _ = env.step(action)

            agent.update(state, action, reward, next_state, terminated)

            state = next_state

            episode_over = terminated or truncated
            
            # agent.render() Avoid rendering on long training runs.

        if print_every and (episode + 1) % print_every == 0:
            print(
                f"Episode {episode}, Average Reward: {np.mean(list(env.return_queue)[-100:])}"
            )

    return agent, env


n_episodes = 6000

# Register the environment so we can create it with gym.make()
gym.register(
    id="gymnasium_env/BusRouting-v0",
    entry_point=BusEnv,
    max_episode_steps=50000,  # Prevent infinite episodes
)

# Create the environment like any built-in environment
env = gym.make("gymnasium_env/BusRouting-v0")
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

# Can adjust training hyperparameters of agent as needed
agent = BusAgent(
    env, 
    discount=0.95,
    learning_rate=0.001,
    buffer_size=100000,
    batch_size=128,
    target_update_freq=5000,
    epsilon_start=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.99999
    )

# Train the agent
agent, env = train(agent, env, n_episodes=n_episodes, print_every=1000)

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_dir = os.path.join(repo_root, "model")
model_path = os.path.join(model_dir, "dqn_agent.pth")

checkpoint = {
    "model_state_dict": agent.main_q.state_dict(),
    "returns": list(env.return_queue),
    "lengths": list(env.length_queue),
}

torch.save(checkpoint, model_path)

print(f"\nTraining complete!")
print(
    f"Final average return (last 100 episodes): {np.mean(list(env.return_queue)[-100:]):.2f}"
)
