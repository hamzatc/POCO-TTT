#!/bin/bash
#SBATCH --job-name=classical_baselines
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=sbatch/logs/classical_baselines_%j.out
#SBATCH --error=sbatch/logs/classical_baselines_%j.err

# Classical Dynamical Systems Baselines
# DMD, HODMD, EDMD, KernelEDMD, SINDy (no GPU needed)

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

# Run classical baselines comparison
yes | python -u main.py -t compare_classical_baselines
