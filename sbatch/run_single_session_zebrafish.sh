#!/bin/bash
#SBATCH --job-name=ss_zebrafish
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=sbatch/logs/single_session_zebrafish_%j.out
#SBATCH --error=sbatch/logs/single_session_zebrafish_%j.err

# Single-Session Baselines on Zebrafish Ahrens Dataset
# Trains POCO, NLinear, MLP, TexFilter, DLinear, TCN on individual sessions

mkdir -p sbatch/logs
module load cuda/12.2.0-fasrc01

cd /n/holystore01/LABS/pehlevan_lab/Lab/hamza/projects/current/POCO-TTT
source .venv/bin/activate

export WANDB_PROJECT=POCO-TTT
export WANDB_ENTITY=neuroai
export PYTHONUNBUFFERED=1

yes | python -u main.py -t single_session_zebrafish_ahrens
