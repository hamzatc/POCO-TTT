#!/bin/bash
#SBATCH --job-name=dmd_test
#SBATCH --partition=kempner
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=sbatch/logs/dmd_test_%j.out
#SBATCH --error=sbatch/logs/dmd_test_%j.err

# DMD Baseline Test
# Classical dynamical systems method (no GPU needed)

# Create logs directory if it doesn't exist
mkdir -p sbatch/logs

# Load modules
module load python/3.10.13-fasrc01

# Activate environment
cd /n/holystore01/LABS/pehlevan_lab/Lab/hamza/projects/current/POCO-TTT
source .venv/bin/activate

# Set wandb environment variables
export WANDB_PROJECT=POCO-TTT
export WANDB_ENTITY=neuroai

# Force unbuffered Python output
export PYTHONUNBUFFERED=1

# Run DMD test experiment
yes | python -u main.py -t dmd_test
