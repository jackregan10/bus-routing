import gymnasium as gym
import numpy as np
import torch
from src.env import BusEnv
from src.agent import BusAgent

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

        if print_every and (episode + 1) % print_every == 0:
            print(
                f"Episode {episode}, Average Reward: {np.mean(list(env.return_queue)[-100:])}"
            )

    return agent, env


n_episodes = 600

# Register the environment so we can create it with gym.make()
gym.register(
    id="gymnasium_env/BusRouting-v0",
    entry_point=BusEnv,
    max_episode_steps=300,  # Prevent infinite episodes
)

# Create the environment like any built-in environment
env = gym.make("gymnasium_env/BusRouting-v0")
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

# Can adjust training hyperparameters of agent as needed
agent = BusAgent(env)

# Train the agent
agent, env = train(agent, env, n_episodes=n_episodes, print_every=20)

torch.save(agent.main_q.state_dict(), "../model/agent.pth")

print(f"\nTraining complete!")
print(
    f"Final average return (last 100 episodes): {np.mean(list(env.return_queue)[-100:]):.2f}"
)
