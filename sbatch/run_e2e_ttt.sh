#!/bin/bash
#SBATCH --job-name=e2e_ttt
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=sbatch/logs/e2e_ttt_%j.out
#SBATCH --error=sbatch/logs/e2e_ttt_%j.err

# E2E-TTT (End-to-End Test-Time Training)
# Full second-order meta-learning with create_graph=True
# Note: Requires more memory due to computation graph storage

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

# Run E2E-TTT multi-species experiment (auto-confirm with yes)
yes | python main.py -t e2e_ttt_multi_species
