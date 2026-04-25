import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from env import BusEnv

# Register the environment so we can create it with gym.make()
gym.register(
    id="gymnasium_env/BusRouting-v0",
    entry_point=BusEnv,
    max_episode_steps=50000,  # Prevent infinite episodes
)

env = gym.make("gymnasium_env/BusRouting-v0")
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=1000)
env = Monitor(env)

current_dir = os.getcwd() 
repo_root = os.path.abspath(os.path.join(current_dir, ".."))
model_dir = os.path.join(repo_root, "model")
model_path = os.path.join(model_dir, "ppo_agent")

model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
)

model.learn(total_timesteps=100_000)
model.save(model_path)