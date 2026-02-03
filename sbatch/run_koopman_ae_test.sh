#!/bin/bash
#SBATCH --job-name=koopman_ae_test
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=sbatch/logs/koopman_ae_test_%j.out
#SBATCH --error=sbatch/logs/koopman_ae_test_%j.err

# Koopman Autoencoder Test
# Deep Koopman baseline (requires GPU)

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

# Run Koopman AE test experiment
yes | python -u main.py -t koopman_ae_test
