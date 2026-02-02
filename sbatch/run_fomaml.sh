#!/bin/bash
#SBATCH --job-name=fomaml
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=sbatch/logs/fomaml_%j.out
#SBATCH --error=sbatch/logs/fomaml_%j.err

# FOMAML (First-Order MAML) Training
# Meta-learning with first-order gradients only

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

# Run FOMAML multi-species experiment (auto-confirm with yes)
yes | python main.py -t fomaml_multi_species
