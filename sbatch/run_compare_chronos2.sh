#!/bin/bash
#SBATCH --job-name=compare_chronos2
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=sbatch/logs/compare_chronos2_%j.out
#SBATCH --error=sbatch/logs/compare_chronos2_%j.err

# Compare Chronos-2 Zero-Shot vs Trained Baselines
# Compares: Chronos2, NLinear, POCO, MLP
# Datasets: celegansflavell, zebrafishahrens_pc, mice

# Create logs directory if it doesn't exist
mkdir -p sbatch/logs

# Load modules
module load cuda/12.2.0-fasrc01

# Change to project directory
cd /n/holystore01/LABS/pehlevan_lab/Lab/hamza/projects/current/POCO-TTT

# Set wandb environment variables
export WANDB_PROJECT=POCO-TTT
export WANDB_ENTITY=neuroai

# Force unbuffered Python output
export PYTHONUNBUFFERED=1

# Set HuggingFace cache directory to avoid quota issues
export HF_HOME=/n/holystore01/LABS/pehlevan_lab/Lab/hamza/.cache/huggingface

# Run comparison experiment using uv (auto-confirm with yes)
yes | uv run python -u main.py -t compare_chronos2_baseline
