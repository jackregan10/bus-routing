# Setup and SLURM Training Instructions

## Create a Python virtual environment

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Install Python dependencies

Install the required packages for the environment and training scripts:

```bash
pip install -r requirements.txt
pip install stable-baselines3
```

## Submit DQN training with SLURM

From the repository root, submit the DQN job:

```bash
sbatch slurm/train_dqn.sh
```

This job script changes into `src/` and runs:

```bash
python3 -u train_dqn.py
```

## Submit PPO training with SLURM

Submit the PPO job similarly:

```bash
sbatch slurm/train_ppo.sh
```

This job script changes into `src/` and runs:

```bash
python3 -u train_ppo.py
```

## Monitor training progress

- SLURM standard output will be written to `slurm-logs/dqn%j.out` and `slurm-logs/ppo%j.out`.
- Errors will be written to `slurm-logs/dqn%j.err` and `slurm-logs/ppo%j.err`.

### Notes

- The DQN script saves the model checkpoint to `model/dqn_agent.pth`.
- The PPO script saves the trained model to `model/ppo_agent`.
- All notebooks are self contained and can run with cpu or cuda
- Agents that have already been trained can be visualized using `notebooks/ppo-single-episode-data-visualization.ipynb` and `notebooks/dqn-single-episode-data-visualization.ipynb`
