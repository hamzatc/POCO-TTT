#!/bin/bash
#SBATCH --job-name=dynamical_horizons
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=sbatch/logs/dynamical_horizons_%j.out
#SBATCH --error=sbatch/logs/dynamical_horizons_%j.err

# Dynamical Systems Methods - Multi-Horizon Comparison
# Compare DMD, KernelEDMD, KoopmanAE, POCO, NLinear across pred_length = [5, 16, 32, 50]

# Create logs directory if it doesn't exist
mkdir -p sbatch/logs

# Load modules
module load cuda/12.2.0-fasrc01

# Activate environment
cd /n/holystore01/LABS/pehlevan_lab/Lab/hamza/projects/current/POCO-TTT
source .venv/bin/activate

# Set wandb environment variables
export WANDB_PROJECT=POCO-TTT
export WANDB_ENTITY=neuroai

# Force unbuffered Python output
export PYTHONUNBUFFERED=1

# Run multi-horizon comparison
yes | python -u main.py -t compare_dynamical_horizons
