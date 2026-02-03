#!/bin/bash
#SBATCH --job-name=chronos2_zeroshot
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=sbatch/logs/chronos2_zeroshot_%j.out
#SBATCH --error=sbatch/logs/chronos2_zeroshot_%j.err

# Chronos-2 Zero-Shot Evaluation
# Pretrained time series foundation model - no training, inference only
# Note: First run will download ~500MB model weights from HuggingFace

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

# Run Chronos-2 zero-shot evaluation using uv (auto-confirm with yes)
yes | uv run python -u main.py -t chronos2_zeroshot
