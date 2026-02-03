#!/bin/bash
#SBATCH --job-name=paper_baselines
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=sbatch/logs/paper_baselines_%j.out
#SBATCH --error=sbatch/logs/paper_baselines_%j.err

# Paper Baselines Training
# Runs NLinear, MLP_L, TexFilter, NetFormer, Latent_PLRNN on celegansflavell and zebrafishahrens_pc
# For comparison with POCO-TTT methods

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

# Run paper baselines experiment (auto-confirm with yes)
yes | python -u main.py -t paper_baselines
