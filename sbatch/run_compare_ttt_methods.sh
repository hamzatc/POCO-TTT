#!/bin/bash
#SBATCH --job-name=compare_ttt
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --output=sbatch/logs/compare_ttt_%j.out
#SBATCH --error=sbatch/logs/compare_ttt_%j.err

# Compare FOMAML vs E2E-TTT
# Runs both methods on the same datasets for comparison

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

# Run comparison experiment (auto-confirm with yes)
yes | python main.py -t compare_ttt_methods
