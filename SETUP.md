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

If your cluster uses a module system, load the appropriate Python and CUDA modules first.

## Verify the installation

Run a quick import check:

```bash
python -c "import gymnasium, torch, numpy; print('OK')"
```

## Prepare the Slurm output directory

Ensure the Slurm log directory exists before submitting jobs:

```bash
mkdir -p slurm-logs
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
- Replace `%j` with the SLURM job ID after submission.

### Notes

- The DQN script saves the model checkpoint to `model/dqn_agent.pth`.
- The PPO script saves the trained model to `model/ppo_agent`.
- If your cluster has a different Python executable or module environment, update the `python3` command in the Slurm scripts accordingly.

