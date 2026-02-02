#!/bin/bash
# Submit all POCO-TTT experiments to Kempner cluster
# Usage: ./sbatch/run_all_experiments.sh

set -e

echo "Submitting POCO-TTT experiments to Kempner cluster..."

# Create logs directory
mkdir -p sbatch/logs

# Submit POCO baseline (standard training)
echo "Submitting POCO baseline..."
JOB_BASELINE=$(sbatch sbatch/run_poco_baseline.sh | awk '{print $4}')
echo "  Job ID: $JOB_BASELINE"

# Submit FOMAML training
echo "Submitting FOMAML..."
JOB_FOMAML=$(sbatch sbatch/run_fomaml.sh | awk '{print $4}')
echo "  Job ID: $JOB_FOMAML"

# Submit E2E-TTT training
echo "Submitting E2E-TTT..."
JOB_E2E=$(sbatch sbatch/run_e2e_ttt.sh | awk '{print $4}')
echo "  Job ID: $JOB_E2E"

# Submit comparison experiment
echo "Submitting comparison experiment..."
JOB_COMPARE=$(sbatch sbatch/run_compare_ttt_methods.sh | awk '{print $4}')
echo "  Job ID: $JOB_COMPARE"

echo ""
echo "All experiments submitted!"
echo "Track progress at: https://wandb.ai/neuroai/POCO-TTT"
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "View logs with: tail -f sbatch/logs/*.out"
